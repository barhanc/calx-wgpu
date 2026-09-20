import type { Shader } from './shader';
import type { WgpuDispatch } from './dispatch';
import { Tensor, DTYPE_BYTESIZE, type DType } from './tensor';

const UNIFORM_BUFFER_USAGE = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const STORAGE_BUFFER_USAGE =
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

/**
 * Execution context that manages GPU buffers and the sequence of recorded kernel dispatches.
 */
export class WgpuExecutionContext {
  readonly #device: GPUDevice;
  readonly #dispatches: WgpuDispatch[] = [];
  readonly #ownedBuffers = new Set<GPUBuffer>();
  #destroyed = false;

  /**
   * Constructs a new WgpuExecutionContext.
   *
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
   * Transfers ownership of a buffer to the context to ensure it remains
   * allocated for the model lifetime and is destroyed on context disposal.
   *
   * @param buffer The GPUBuffer to take ownership of.
   */
  #ownBuffer(buffer: GPUBuffer): void {
    this.#ownedBuffers.add(buffer);
  }

  /**
   * Destroys all owned buffers and releases resources.
   */
  destroy(): void {
    if (this.#destroyed) {
      return;
    }
    this.#destroyed = true;
    for (const buffer of this.#ownedBuffers) {
      buffer.destroy();
    }
    this.#ownedBuffers.clear();
    this.#dispatches.length = 0;
  }

  // =========================================================================
  // Buffer Allocation
  // =========================================================================

  /**
   * Creates an empty WebGPU storage buffer whose lifetime is managed by this context.
   *
   * @param size Size in bytes to allocate.
   * @returns The newly allocated and owned GPUBuffer with STORAGE | COPY_SRC | COPY_DST usage.
   */
  storageBuffer(size: number): GPUBuffer {
    if (this.#destroyed) {
      throw new Error('WgpuExecutionContext is destroyed');
    }
    const alignedSize = Math.max(4, Math.ceil(size / 4) * 4);
    const buffer = this.#device.createBuffer({ size: alignedSize, usage: STORAGE_BUFFER_USAGE });
    this.#ownBuffer(buffer);
    return buffer;
  }

  /**
   * Creates and populates a WebGPU uniform buffer whose lifetime is managed by this context.
   *
   * @param data Binary data to copy into the uniform buffer.
   * @returns The newly allocated and owned GPUBuffer.
   */
  uniformBuffer(data: ArrayBufferView | ArrayBuffer): GPUBuffer {
    if (this.#destroyed) {
      throw new Error('WgpuExecutionContext is destroyed');
    }
    const byteLength = data.byteLength;
    const alignedSize = Math.max(16, Math.ceil(byteLength / 4) * 4);
    const buffer = this.#device.createBuffer({ size: alignedSize, usage: UNIFORM_BUFFER_USAGE });

    if (data instanceof ArrayBuffer) {
      this.#device.queue.writeBuffer(buffer, 0, data, 0, byteLength);
    } else {
      this.#device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, byteLength);
    }

    this.#ownBuffer(buffer);
    return buffer;
  }

  /**
   * Creates a Tensor view over a GPU buffer.
   *
   * Buffer ownership behavior:
   * 1. **`buffer` omitted**: The context allocates a fresh `GPUBuffer` using {@link storageBuffer}.
   *    This buffer is tracked and owned by the context, and will be automatically destroyed when
   *    {@link destroy} is called on this context.
   * 2. **`buffer` provided**: The context wraps the provided external `GPUBuffer` (at `byteOffset`)
   *    without taking ownership. The buffer's lifetime is managed externally by the caller and will
   *    NOT be tracked or destroyed by this context.
   *
   * @param dtype Element data type of the tensor.
   * @param shape Tensor dimensions.
   * @param buffer Optional existing GPUBuffer to wrap. If omitted, the context allocates and owns a new buffer.
   * @param byteOffset Optional byte offset in the buffer (defaults to 0, must be a multiple of 4).
   * @returns A new Tensor view.
   */
  tensor(
    dtype: DType,
    shape: readonly number[],
    buffer?: GPUBuffer,
    byteOffset: number = 0
  ): Tensor {
    if (this.#destroyed) {
      throw new Error('WgpuExecutionContext is destroyed');
    }
    if (buffer !== undefined) {
      return new Tensor(dtype, shape, this.#device, buffer, byteOffset);
    }
    const numel = shape.reduce((a, b) => a * b, 1);
    const byteLength = numel * DTYPE_BYTESIZE[dtype];
    const newBuffer = this.storageBuffer(byteLength);

    return new Tensor(dtype, shape, this.#device, newBuffer);
  }

  // =========================================================================
  // Dispatch Recording & Execution
  // =========================================================================

  /**
   * Records a compute dispatch into the context execution queue.
   *
   * @param dispatch The dispatch descriptor to record.
   */
  addDispatch(dispatch: WgpuDispatch): void {
    if (this.#destroyed) {
      throw new Error('WgpuExecutionContext is destroyed');
    }
    this.#dispatches.push(dispatch);
  }

  /**
   * Records a compute shader dispatch into this context.
   *
   * @param shader The compute shader to execute.
   * @param args Positional arguments expected by the operator.
   * @returns This context instance.
   */
  record<TArgs extends readonly unknown[]>(shader: Shader<TArgs>, args: TArgs): this {
    if (this.#destroyed) {
      throw new Error('WgpuExecutionContext is destroyed');
    }
    shader.recordIn(this, args);
    return this;
  }

  /**
   * Encodes all recorded dispatches into a single command buffer and submits to the GPU queue.
   */
  submit(): void {
    if (this.#destroyed) {
      throw new Error('WgpuExecutionContext is destroyed');
    }
    if (this.#dispatches.length === 0) {
      return;
    }

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
   * while keeping owned buffers intact.
   */
  reset(): void {
    if (this.#destroyed) {
      throw new Error('WgpuExecutionContext is destroyed');
    }
    this.#dispatches.length = 0;
  }
}
