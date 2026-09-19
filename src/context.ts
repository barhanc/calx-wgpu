import type { Kernel } from './kernel';
import type { WgpuDispatch } from './dispatch';
import { Tensor, type DType, type TypedArray } from './tensor';

/**
 * Execution context that manages GPU tensors, uniform buffers,
 * and the sequence of recorded kernel dispatches.
 */
export class WgpuExecutionContext {
  readonly #device: GPUDevice;
  readonly #tensors = new Map<PropertyKey, Tensor>();
  readonly #scalars = new Map<PropertyKey, number>();
  readonly #dispatches: WgpuDispatch[] = [];
  readonly #ownedBuffers = new Set<GPUBuffer>();

  /**
   * Constructs a new WgpuExecutionContext.
   * @param device The WebGPU device instance.
   */
  constructor(device: GPUDevice) {
    this.#device = device;
  }

  /** The WebGPU device used by this context. */
  get device(): GPUDevice {
    return this.#device;
  }

  /**
   * Retrieves a tensor value by its context value ID.
   * @param id Identifier of the value in the context.
   * @returns The Tensor instance.
   */
  getTensor(id: PropertyKey): Tensor {
    const t = this.#tensors.get(id);
    if (!t) {
      throw new Error(`WgpuExecutionContext: Tensor with ID ${String(id)} not found in context`);
    }
    return t;
  }

  /**
   * Sets or replaces a tensor value in the context.
   * @param id Identifier of the value.
   * @param t The Tensor instance.
   * @returns This context for chaining.
   */
  setTensor(id: PropertyKey, t: Tensor): this {
    if (t.device !== this.#device) {
      throw new Error('WgpuExecutionContext: Tensor device mismatch');
    }
    this.#tensors.set(id, t);
    return this;
  }

  /**
   * Retrieves a scalar constant by its context value ID.
   * @param id Identifier of the scalar in the context.
   * @returns The numeric value.
   */
  getScalar(id: PropertyKey): number {
    const s = this.#scalars.get(id);
    if (s === undefined) {
      throw new Error(`WgpuExecutionContext: Scalar with ID ${String(id)} not found in context`);
    }
    return s;
  }

  /**
   * Sets a scalar constant in the context.
   * @param id Identifier of the scalar.
   * @param val The numeric value.
   * @returns This context for chaining.
   */
  setScalar(id: PropertyKey, val: number): this {
    this.#scalars.set(id, val);
    return this;
  }

  /**
   * Creates a new Tensor whose underlying GPUBuffer is automatically tracked
   * and released when this execution context is destroyed.
   *
   * @param dtype Data type of elements.
   * @param shape Dimensions of the tensor (rank <= 8).
   * @param src Optional initial host data or existing GPUBuffer to wrap.
   * @returns A newly created context-owned Tensor instance.
   */
  tensor(dtype: DType, shape: readonly number[], src?: GPUBuffer | TypedArray): Tensor {
    const t = new Tensor(dtype, shape, this.#device, src);
    this.ownBuffer(t.buffer);
    return t;
  }

  /**
   * Creates an empty WebGPU storage buffer whose lifetime is managed by this context.
   *
   * @param size Size in bytes to allocate.
   * @returns The newly allocated and owned GPUBuffer with STORAGE | COPY_SRC | COPY_DST usage.
   */
  storageBuffer(size: number): GPUBuffer {
    const alignedSize = Math.max(4, Math.ceil(size / 4) * 4);
    const buffer = this.#device.createBuffer({
      size: alignedSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    this.ownBuffer(buffer);
    return buffer;
  }

  /**
   * Creates and populates a WebGPU uniform buffer whose lifetime is managed
   * by this context.
   *
   * @param data Binary data to copy into the uniform buffer.
   * @returns The newly allocated and owned GPUBuffer.
   */
  uniformBuffer(data: ArrayBufferView | ArrayBuffer): GPUBuffer {
    const byteLength = data.byteLength;
    const alignedSize = Math.max(16, Math.ceil(byteLength / 4) * 4);

    const buffer = this.#device.createBuffer({
      size: alignedSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    if (data instanceof ArrayBuffer) {
      this.#device.queue.writeBuffer(buffer, 0, data, 0, byteLength);
    } else {
      this.#device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, byteLength);
    }

    this.ownBuffer(buffer);
    return buffer;
  }

  /**
   * Records a compute dispatch into the context execution queue.
   * @param dispatch The dispatch descriptor to record.
   */
  addDispatch(dispatch: WgpuDispatch): void {
    this.#dispatches.push(dispatch);
  }

  /**
   * Transfers ownership of a buffer to the context to ensure it remains
   * allocated for the model lifetime and is destroyed on context disposal.
   * @param buffer The GPUBuffer to take ownership of.
   */
  ownBuffer(buffer: GPUBuffer): void {
    this.#ownedBuffers.add(buffer);
  }

  /**
   * Dispatches a compute kernel into this context.
   *
   * @param kernel The compute kernel to execute.
   * @param args Positional argument IDs referring to entries in this context.
   * @returns This context for chaining.
   */
  dispatch<TArgs extends readonly PropertyKey[]>(kernel: Kernel<TArgs>, args: TArgs): this {
    kernel.dispatchIn(this, args);
    return this;
  }

  /**
   * Encodes all recorded dispatches into a single command buffer and submits to the GPU queue.
   */
  submit(): void {
    if (this.#dispatches.length === 0) return;

    const encoder = this.#device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    for (const dispatch of this.#dispatches) {
      pass.setPipeline(dispatch.pipeline);
      pass.setBindGroup(0, dispatch.bindGroup);
      pass.dispatchWorkgroups(
        dispatch.workgroupCountX,
        dispatch.workgroupCountY,
        dispatch.workgroupCountZ ?? 1
      );
    }

    pass.end();
    this.#device.queue.submit([encoder.finish()]);
  }

  /**
   * Clears all recorded dispatches to prepare for a fresh recording pass
   * while keeping registered tensors, scalars, and owned buffers intact.
   */
  reset(): void {
    this.#dispatches.length = 0;
  }

  /**
   * Destroys all owned buffers and releases resources.
   */
  destroy(): void {
    for (const buffer of this.#ownedBuffers) {
      buffer.destroy();
    }
    this.#ownedBuffers.clear();
    this.#dispatches.length = 0;
    this.#tensors.clear();
    this.#scalars.clear();
  }
}
