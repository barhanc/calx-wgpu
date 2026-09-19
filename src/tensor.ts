/**
 * Supported tensor data types in WebGPU storage buffers.
 */
export type DType =
  'float32' | 'float16' | 'int32' | 'int8' | 'uint8' | 'bool' | 'float64' | 'int64';

/**
 * Supported typed arrays that can be uploaded to a Tensor.
 */
export type TypedArray =
  Float32Array | Int32Array | Int8Array | Uint8Array | Uint16Array | Float64Array | BigInt64Array;

/**
 * Maximum tensor rank supported by WebGPU kernels (std140 layout limit).
 */
export const MAX_NDIM = 8;

// prettier-ignore
export const DTYPE_BYTESIZE: Record<DType, number> = {
  float32: 4, float16: 2,
  int32:   4, int8:    1,
  uint8:   1, bool:    1,
  float64: 8, int64:   8,
} as const;

const STAGING_BUFFER_USAGE = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
const STORAGE_BUFFER_USAGE =
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

/**
 * A WebGPU-backed Tensor instance.
 *
 * Encapsulates an allocated `GPUBuffer` in device VRAM along with its metadata
 * (`dtype`, `shape`, `numel`). Use {@link Tensor.getData} to read raw bytes
 * back into host memory, {@link Tensor.setData} to copy data into it,
 * or {@link Tensor.destroy} to release GPU memory.
 */
export class Tensor {
  readonly #dtype: DType;
  readonly #numel: number;
  readonly #shape: readonly number[];
  readonly #device: GPUDevice;
  readonly #buffer: GPUBuffer;
  #destroyed = false;

  constructor(
    dtype: DType,
    shape: readonly number[],
    device: GPUDevice,
    src?: GPUBuffer | TypedArray
  ) {
    if (shape.length > MAX_NDIM) {
      throw new Error(`Tensor rank ${shape.length} exceeds maximum rank of ${MAX_NDIM}`);
    }
    if (shape.some((dim) => dim <= 0 || !Number.isInteger(dim))) {
      throw new Error('Tensor dimensions must be positive integers');
    }

    this.#dtype = dtype;
    this.#shape = shape;
    this.#numel = shape.reduce((a, b) => a * b, 1);
    this.#device = device;

    const byteLength = this.#numel * DTYPE_BYTESIZE[dtype];
    const alignedSize = Math.max(16, Math.ceil(byteLength / 4) * 4);

    if (src === undefined) {
      this.#buffer = device.createBuffer({ size: alignedSize, usage: STORAGE_BUFFER_USAGE });
      return;
    }

    if (src instanceof GPUBuffer) {
      if (src.size < byteLength) {
        throw new Error(`GPUBuffer size (${src.size}B) < required tensor size (${byteLength}B)`);
      }

      if ((src.usage & GPUBufferUsage.STORAGE) === 0) {
        throw new Error('GPUBuffer must have GPUBufferUsage.STORAGE flag');
      }

      this.#buffer = src;
      return;
    }

    if (src.byteLength !== byteLength) {
      throw new Error(`Source bytes (${src.byteLength}B) !== tensor size (${byteLength}B)`);
    }

    this.#buffer = device.createBuffer({ size: alignedSize, usage: STORAGE_BUFFER_USAGE });
    device.queue.writeBuffer(this.#buffer, 0, src.buffer, src.byteOffset, byteLength);
  }

  /** The element data type of the tensor. */
  get dtype(): DType {
    return this.#dtype;
  }

  /** Total number of elements across all dimensions. */
  get numel(): number {
    return this.#numel;
  }

  /** The dimensions of the tensor. */
  get shape(): readonly number[] {
    return this.#shape;
  }

  /** The WebGPU device that owns this tensor's buffer. */
  get device(): GPUDevice {
    return this.#device;
  }

  /** The underlying WebGPU storage buffer in VRAM. */
  get buffer(): GPUBuffer {
    return this.#buffer;
  }

  /**
   * The total byte size of the tensor data.
   */
  get nbytes(): number {
    return this.#numel * DTYPE_BYTESIZE[this.#dtype];
  }

  /**
   * Copies data from a host TypedArray or existing GPUBuffer into this tensor's
   * storage buffer.
   *
   * @param src Source TypedArray or GPUBuffer to copy from.
   * @returns This tensor instance.
   * @throws {Error} If source byte size does not match this tensor's byte length.
   */
  setData(src: GPUBuffer | TypedArray): this {
    if (this.#destroyed) {
      throw new Error('Tensor is destroyed');
    }

    const nbytes = this.nbytes;

    if (src instanceof GPUBuffer) {
      if (src.size < nbytes) {
        throw new Error(`GPUBuffer size (${src.size}B) < tensor size (${nbytes}B)`);
      }

      if ((src.usage & GPUBufferUsage.COPY_SRC) === 0) {
        throw new Error('GPUBuffer must have GPUBufferUsage.COPY_SRC flag');
      }

      const encoder = this.#device.createCommandEncoder();
      encoder.copyBufferToBuffer(src, 0, this.#buffer, 0, nbytes);
      this.#device.queue.submit([encoder.finish()]);
      return this;
    }

    if (src.byteLength !== nbytes) {
      throw new Error(`Source bytes (${src.byteLength}B) !== tensor byte size (${nbytes}B)`);
    }
    this.#device.queue.writeBuffer(this.#buffer, 0, src.buffer, src.byteOffset, nbytes);
    return this;
  }

  /**
   * Reads raw bytes back from the GPU storage buffer into host memory.
   *
   * @returns Raw byte buffer containing the tensor data.
   */
  async getData(): Promise<ArrayBuffer> {
    if (this.#destroyed) {
      throw new Error('Tensor is destroyed');
    }

    const nbytes = this.nbytes;
    const alignedSize = Math.max(16, Math.ceil(nbytes / 4) * 4);

    const encoder = this.#device.createCommandEncoder();
    const staging = this.#device.createBuffer({ size: alignedSize, usage: STAGING_BUFFER_USAGE });

    try {
      encoder.copyBufferToBuffer(this.#buffer, 0, staging, 0, alignedSize);
      this.#device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      return staging.getMappedRange(0, nbytes).slice(0);
    } finally {
      staging.destroy();
    }
  }

  /**
   * Destroys and releases the underlying GPU buffer.
   */
  destroy(): void {
    if (this.#destroyed) return;
    this.#destroyed = true;
    this.#buffer.destroy();
  }
}
