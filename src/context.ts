import type { Tensor } from './tensor';
import type { WgpuDispatch } from './dispatch';

/**
 * Execution context that manages GPU tensors, uniform buffers,
 * and the sequence of recorded kernel dispatches.
 */
export class WgpuExecutionContext {
  readonly #device: GPUDevice;
  readonly #tensors = new Map<number, Tensor>();
  readonly #scalars = new Map<number, number>();
  readonly #dispatches: WgpuDispatch[] = [];
  readonly #ownedBuffers: GPUBuffer[] = [];

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
   * @param id Integer ID of the value in the context.
   * @returns The Tensor instance.
   */
  getTensor(id: number): Tensor {
    const t = this.#tensors.get(id);
    if (!t) {
      throw new Error(`WgpuExecutionContext: Tensor with ID ${id} not found in context`);
    }
    return t;
  }

  /**
   * Sets or replaces a tensor value in the context.
   * @param id Integer ID of the value.
   * @param t The Tensor instance.
   */
  setTensor(id: number, t: Tensor): void {
    if (t.device !== this.#device) {
      throw new Error('WgpuExecutionContext: Tensor device mismatch');
    }
    this.#tensors.set(id, t);
  }

  /**
   * Retrieves a scalar constant by its context value ID.
   * @param id Integer ID of the scalar in the context.
   * @returns The numeric value.
   */
  getScalar(id: number): number {
    const s = this.#scalars.get(id);
    if (s === undefined) {
      throw new Error(`WgpuExecutionContext: Scalar with ID ${id} not found in context`);
    }
    return s;
  }

  /**
   * Sets a scalar constant in the context.
   * @param id Integer ID of the scalar.
   * @param val The numeric value.
   */
  setScalar(id: number, val: number): void {
    this.#scalars.set(id, val);
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
    this.#ownedBuffers.push(buffer);
  }

  /**
   * Encodes all recorded dispatches into a single command buffer and submits to the GPU.
   */
  execute(): void {
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
   * Destroys all owned buffers and releases resources.
   */
  destroy(): void {
    for (const buffer of this.#ownedBuffers) {
      buffer.destroy();
    }
    this.#ownedBuffers.length = 0;
    this.#dispatches.length = 0;
    this.#tensors.clear();
    this.#scalars.clear();
  }
}
