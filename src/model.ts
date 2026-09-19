import { WgpuExecutionContext } from './context';
import type { Kernel } from './kernel';
import { kernels } from './kernels';
import { parsePte, type ParsedProgram, type ParsedVkValue } from './parser';
import { Tensor, type DType } from './tensor';

/**
 * Any value that can be passed as an input argument or returned from a model method.
 */
export type ModelValue = Tensor | number | boolean | string | null | readonly ModelValue[];

export type ModelInput = ModelValue;
export type ModelOutput = ModelValue;

/**
 * Type guard to check if a value is a Phlox Tensor.
 *
 * @param val Candidate value.
 * @returns True if value is an instance of Tensor.
 */
function isTensor(val: unknown): val is Tensor {
  return val instanceof Tensor;
}

/**
 * Normalizes ExecuTorch operator target names to our internal registry keys.
 * Handles variations such as 'aten.add.Tensor', 'aten::add.Tensor',
 * 'aten.mm.default', and 'aten::mm.default'.
 *
 * @param opName Operator target name from the delegate graph.
 * @returns The matching Kernel if registered.
 */
function resolveKernel(opName: string): Kernel | undefined {
  const normalized = opName.replace('.', '::');
  if (normalized === 'aten::add.Tensor' || opName === 'aten.add.Tensor') {
    return kernels.add as Kernel;
  }
  if (normalized === 'aten::mm.default' || opName === 'aten.mm.default') {
    return kernels.mm as Kernel;
  }
  return undefined;
}

/**
 * Internal method execution plan state.
 */
type MethodPlan = {
  readonly name: string;
  readonly inputIds: readonly number[];
  readonly outputIds: readonly number[];
  readonly values: readonly ParsedVkValue[];
  readonly context: WgpuExecutionContext;
  readonly sharedBuffers: Map<number, GPUBuffer>;
  readonly outputTensors: Map<number, Tensor>;
};

/**
 * Represents a loaded ExecuTorch `.pte` model ready for WebGPU inference.
 * Supports executing arbitrary exported methods (e.g. 'forward', 'encode', 'decode').
 */
export class Model {
  readonly #device: GPUDevice;
  readonly #program: ParsedProgram;
  readonly #plans = new Map<string, MethodPlan>();
  #disposed = false;

  private constructor(device: GPUDevice, program: ParsedProgram) {
    this.#device = device;
    this.#program = program;
    this.#initializePlans();
  }

  /**
   * Loads an ExecuTorch `.pte` model from a binary buffer.
   *
   * @param device The WebGPU GPUDevice instance.
   * @param pteData Binary buffer or Uint8Array of the `.pte` model file.
   * @returns A compiled, ready-to-run Model instance.
   */
  static load(device: GPUDevice, pteData: ArrayBuffer | Uint8Array): Model {
    const bytes = pteData instanceof Uint8Array ? pteData : new Uint8Array(pteData);
    const program = parsePte(bytes);
    return new Model(device, program);
  }

  /**
   * The list of all exported method names available in this model.
   */
  get methodNames(): readonly string[] {
    return Array.from(this.#program.methods.keys());
  }

  /**
   * Initialises and pre-allocates execution plans, GPU buffers, constants, and kernel dispatches.
   */
  #initializePlans(): void {
    for (const [methodName, method] of this.#program.methods) {
      const { delegate } = method;
      const ctx = new WgpuExecutionContext(this.#device);

      // Pass 1: Compute maximum allocation sizes for shared memory objects (mem_obj_id >= 0)
      const sharedSizes = new Map<number, number>();
      for (const val of delegate.values) {
        if (val.kind === 'tensor' && val.memObjId >= 0) {
          const numel = val.dims.reduce((acc, d) => acc * d, 1);
          const elemSize = val.dtype === 'float32' ? 4 : 1;
          const byteSize = Math.max(numel * elemSize, 4);
          const currentMax = sharedSizes.get(val.memObjId) ?? 0;
          if (byteSize > currentMax) {
            sharedSizes.set(val.memObjId, byteSize);
          }
        }
      }

      // Pass 2: Allocate shared GPU storage buffers
      const sharedBuffers = new Map<number, GPUBuffer>();
      for (const [memId, size] of sharedSizes) {
        const alignedSize = (size + 3) & ~3;
        const buf = this.#device.createBuffer({
          size: alignedSize,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        ctx.ownBuffer(buf);
        sharedBuffers.set(memId, buf);
      }

      // Pass 3: Create tensors and populate context values
      const outputTensors = new Map<number, Tensor>();
      for (let i = 0; i < delegate.values.length; i++) {
        const val = delegate.values[i];
        if (val.kind === 'tensor') {
          let buffer: GPUBuffer;
          if (val.memObjId >= 0) {
            buffer = sharedBuffers.get(val.memObjId)!;
          } else {
            const numel = val.dims.reduce((acc, d) => acc * d, 1);
            const elemSize = val.dtype === 'float32' ? 4 : 1;
            const alignedSize = Math.max((numel * elemSize + 3) & ~3, 4);
            buffer = this.#device.createBuffer({
              size: alignedSize,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            ctx.ownBuffer(buffer);
          }

          const t = new Tensor(val.dtype as DType, val.dims, this.#device, buffer);
          ctx.setTensor(i, t);

          // Upload constant weight data if present
          if (val.constantId >= 0 && val.constantId < delegate.constants.length) {
            const constMeta = delegate.constants[val.constantId];
            const rawConst = delegate.constantData.slice(
              constMeta.offset,
              constMeta.offset + constMeta.length
            );
            this.#device.queue.writeBuffer(
              buffer,
              0,
              rawConst.buffer,
              rawConst.byteOffset,
              rawConst.byteLength
            );
          }

          if (delegate.outputIds.includes(i)) {
            outputTensors.set(i, t);
          }
        } else if (val.kind === 'scalar') {
          if (typeof val.value === 'number') {
            ctx.setScalar(i, val.value);
          }
        }
      }

      // Pass 4: Build operator dispatch chain
      for (const op of delegate.chain) {
        const kernel = resolveKernel(op.name);
        if (!kernel) {
          throw new Error(`Unsupported ExecuTorch WebGPU operator: '${op.name}'`);
        }
        kernel.attachTo(ctx, op.args);
      }

      this.#plans.set(methodName, {
        name: methodName,
        inputIds: delegate.inputIds,
        outputIds: delegate.outputIds,
        values: delegate.values,
        context: ctx,
        sharedBuffers,
        outputTensors,
      });
    }
  }

  /**
   * Executes an exported method by name.
   *
   * @param methodName The method name to execute (e.g. 'forward').
   * @param inputs Ordered list of input arguments.
   * @returns Array of output values produced by the method.
   */
  async execute(methodName: string, inputs: readonly ModelInput[]): Promise<ModelOutput[]> {
    if (this.#disposed) {
      throw new Error('Model has been disposed and cannot be executed');
    }

    const plan = this.#plans.get(methodName);
    if (!plan) {
      throw new Error(`Method '${methodName}' not found in loaded model`);
    }

    if (inputs.length !== plan.inputIds.length) {
      throw new Error(
        `Method '${methodName}' expects ${plan.inputIds.length} inputs, got ${inputs.length}`
      );
    }

    // Bind inputs to context slots
    for (let idx = 0; idx < inputs.length; idx++) {
      const input = inputs[idx];
      const targetId = plan.inputIds[idx];
      const expectedVal = plan.values[targetId];

      if (isTensor(input)) {
        if (expectedVal.kind !== 'tensor') {
          throw new Error(`Input ${idx} expected scalar but received Tensor`);
        }
        const targetTensor = plan.context.getTensor(targetId);

        // Copy host input data to GPU buffer
        const encoder = this.#device.createCommandEncoder();
        encoder.copyBufferToBuffer(input.buffer, 0, targetTensor.buffer, 0, input.buffer.size);
        this.#device.queue.submit([encoder.finish()]);
      } else if (typeof input === 'number') {
        plan.context.setScalar(targetId, input);
      }
    }

    // Submit WebGPU compute pass
    plan.context.execute();

    // Readback outputs
    const outputs: ModelOutput[] = [];
    for (const outId of plan.outputIds) {
      const outVal = plan.values[outId];
      if (outVal.kind === 'tensor') {
        const outTensor = plan.outputTensors.get(outId)!;
        outputs.push(outTensor);
      } else if (outVal.kind === 'scalar') {
        outputs.push(plan.context.getScalar(outId));
      } else {
        outputs.push(null);
      }
    }

    return outputs;
  }

  /**
   * Convenience shorthand for executing the default 'forward' method.
   *
   * @param inputs Input arguments passed to the forward method.
   * @returns Output values produced by the forward method.
   */
  async forward(...inputs: readonly ModelInput[]): Promise<ModelOutput[]> {
    return this.execute('forward', inputs);
  }

  /**
   * Releases all GPU buffers, bind groups, and execution resources owned by this model.
   */
  dispose(): void {
    if (this.#disposed) return;
    for (const plan of this.#plans.values()) {
      plan.context.destroy();
      for (const buf of plan.sharedBuffers.values()) {
        buf.destroy();
      }
    }
    this.#plans.clear();
    this.#disposed = true;
  }
}
