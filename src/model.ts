import { WgpuExecutionContext } from './context';
import type { Shader } from './shader';
import { shaders } from './shaders';
import { parsePte, DelegateGraph } from './parser';
import { Tensor } from './tensor';

export type ModelValue = Tensor | number | boolean | string | null | readonly ModelValue[];
export type ModelInput = ModelValue;
export type ModelOutput = ModelValue;

type MethodPlan = {
  readonly name: string;
  readonly inputIds: readonly number[];
  readonly outputIds: readonly number[];
  readonly context: WgpuExecutionContext;
  readonly outputTensors: Map<number, Tensor>;
};

/**
 * Represents a loaded ExecuTorch `.pte` model ready for WebGPU inference.
 */
export class Model {
  readonly #device: GPUDevice;
  readonly #plans = new Map<string, MethodPlan>();
  #disposed = false;

  private constructor(device: GPUDevice) {
    this.#device = device;
  }

  /**
   * Loads an ExecuTorch `.pte` model from a binary buffer.
   * @param device The WebGPU GPUDevice instance.
   * @param pteData Binary buffer or Uint8Array of the `.pte` model file.
   * @returns A compiled, ready-to-run Model instance.
   */
  static load(device: GPUDevice, pteData: ArrayBuffer | Uint8Array): Model {
    const bytes = pteData instanceof Uint8Array ? pteData : new Uint8Array(pteData);
    const parsed = parsePte(bytes);
    const model = new Model(device);

    for (let i = 0; i < parsed.methodNames.length; i++) {
      const name = parsed.methodNames[i];
      const delegate = parsed.getDelegate(i);
      model.#buildPlan(name, delegate);
    }
    return model;
  }

  /**
   * The list of all exported method names available in this model.
   */
  get methodNames(): readonly string[] {
    return Array.from(this.#plans.keys());
  }

  #buildPlan(methodName: string, delegate: DelegateGraph): void {
    const ctx = new WgpuExecutionContext(this.#device);

    // Pass 1: Compute maximum allocation sizes for shared memory objects (mem_obj_id >= 0)
    const sharedSizes = new Map<number, number>();
    for (let i = 0; i < delegate.valuesCount; i++) {
      const val = delegate.getValue(i);
      if (val.kind === 'tensor' && val.memObjId >= 0) {
        const current = sharedSizes.get(val.memObjId) ?? 0;
        if (val.byteSize > current) sharedSizes.set(val.memObjId, val.byteSize);
      }
    }

    // Pass 2: Allocate shared GPU storage buffers
    const sharedBuffers = new Map<number, GPUBuffer>();
    for (const [memId, size] of sharedSizes) {
      sharedBuffers.set(memId, ctx.storageBuffer(size));
    }

    // Pass 3: Create tensors and populate context values
    const outputTensors = new Map<number, Tensor>();
    const outputIds = delegate.outputIds;

    for (let i = 0; i < delegate.valuesCount; i++) {
      const val = delegate.getValue(i);
      if (val.kind === 'tensor') {
        const buffer =
          val.memObjId >= 0 ? sharedBuffers.get(val.memObjId)! : ctx.storageBuffer(val.byteSize);

        ctx.setTensor(i, val.dtype, val.dims, buffer);
        const t = ctx.getTensor(i);

        const constData = delegate.getConstant(val.constantId);
        if (constData) t.setData(constData);

        if (outputIds.includes(i)) outputTensors.set(i, t);
      } else if (val.kind === 'scalar') {
        ctx.setScalar(i, val.value);
      }
    }

    // Pass 4: Build operator dispatch chain
    for (let i = 0; i < delegate.opsCount; i++) {
      const op = delegate.getOp(i);
      const shader = Object.values(shaders).find((s) => s.name === op.name) as Shader | undefined;
      if (!shader) throw new Error(`Unsupported ExecuTorch operator: '${op.name}'`);
      shader.recordIn(ctx, op.args);
    }

    this.#plans.set(methodName, {
      name: methodName,
      inputIds: delegate.inputIds,
      outputIds,
      context: ctx,
      outputTensors,
    });
  }

  /**
   * Executes an exported method by name.
   * @param methodName The method name to execute (e.g. 'forward').
   * @param inputs Ordered list of input arguments.
   * @returns Array of output values produced by the method.
   */
  async execute(methodName: string, inputs: readonly ModelInput[] = []): Promise<ModelOutput[]> {
    if (this.#disposed) throw new Error('Model is disposed');

    const plan = this.#plans.get(methodName);
    if (!plan) throw new Error(`Method '${methodName}' not found in loaded model`);
    if (inputs.length !== plan.inputIds.length) {
      throw new Error(`Method '${methodName}' expects ${plan.inputIds.length} inputs`);
    }

    for (let idx = 0; idx < inputs.length; idx++) {
      const input = inputs[idx];
      const targetId = plan.inputIds[idx];
      if (input instanceof Tensor) {
        plan.context.getTensor(targetId).setData(input.buffer);
      } else if (typeof input === 'number') {
        plan.context.setScalar(targetId, input);
      }
    }

    plan.context.submit();

    const outputs: ModelOutput[] = [];
    for (const outId of plan.outputIds) {
      outputs.push(plan.outputTensors.get(outId) ?? plan.context.getScalar(outId));
    }
    return outputs;
  }

  /**
   * Releases all GPU buffers, bind groups, and execution resources owned by this model.
   */
  dispose(): void {
    if (this.#disposed) return;
    for (const plan of this.#plans.values()) plan.context.destroy();
    this.#plans.clear();
    this.#disposed = true;
  }
}
