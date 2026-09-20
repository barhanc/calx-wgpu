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

/**
 * A non-owning WebGPU-backed Tensor view.
 *
 * Encapsulates an interpretation of a slice of an allocated `GPUBuffer` in
 * device VRAM along with its metadata (`dtype`, `shape`, `numel`, `byteLength`,
 * `byteOffset`). Use {@link Tensor.getData} to read raw bytes back into host
 * memory and {@link Tensor.setData} to copy data into it. The underlying
 * `GPUBuffer` lifetime is managed externally.
 */
export class Tensor {
  readonly #dtype: DType;
  readonly #shape: readonly number[];
  readonly #numel: number;
  readonly #buffer: GPUBuffer;
  readonly #device: GPUDevice;
  readonly #byteOffset: number;

  constructor(
    dtype: DType,
    shape: readonly number[],
    device: GPUDevice,
    buffer: GPUBuffer,
    byteOffset: number = 0
  ) {
    if (shape.length > MAX_NDIM) {
      throw new Error(`Tensor rank ${shape.length} exceeds maximum rank of ${MAX_NDIM}`);
    }
    if (shape.some((dim) => dim <= 0 || !Number.isInteger(dim))) {
      throw new Error('Tensor dimensions must be positive integers');
    }
    if (byteOffset < 0 || !Number.isInteger(byteOffset)) {
      throw new Error('byteOffset must be a non-negative integer');
    }
    if (byteOffset % 4 !== 0) {
      throw new Error(`byteOffset (${byteOffset}) must be a multiple of 4 bytes`);
    }

    this.#dtype = dtype;
    this.#shape = shape;
    this.#numel = shape.reduce((a, b) => a * b, 1);
    this.#buffer = buffer;
    this.#device = device;
    this.#byteOffset = byteOffset;

    const requiredBytes = byteOffset + this.#numel * DTYPE_BYTESIZE[dtype];
    if (buffer.size < requiredBytes) {
      throw new Error(`GPUBuffer size (${buffer.size}B) < required (${requiredBytes}B)`);
    }
    if ((buffer.usage & GPUBufferUsage.STORAGE) === 0) {
      throw new Error('GPUBuffer must have GPUBufferUsage.STORAGE flag');
    }
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

  /** The underlying WebGPU storage buffer in VRAM. */
  get buffer(): GPUBuffer {
    return this.#buffer;
  }

  /** The WebGPU device that owns this tensor's buffer. */
  get device(): GPUDevice {
    return this.#device;
  }

  /** Byte offset into the underlying buffer where this tensor begins. */
  get byteOffset(): number {
    return this.#byteOffset;
  }

  /**
   * The total byte size of the tensor data slice.
   */
  get byteLength(): number {
    return this.#numel * DTYPE_BYTESIZE[this.#dtype];
  }

  /**
   * Copies data from a host TypedArray or another GPUBuffer into this tensor's
   * slice in the underlying storage buffer.
   *
   * @param src Source TypedArray or GPUBuffer to copy from.
   * @param srcOffset Optional byte offset into src to begin copying from
   * (default: 0).
   * @returns This tensor instance.
   * @throws {Error} If offset or size boundaries exceed buffer limits.
   */
  setData(src: GPUBuffer | TypedArray, srcOffset: number = 0): this {
    if (srcOffset < 0 || !Number.isInteger(srcOffset)) {
      throw new Error('srcOffset must be a non-negative integer');
    }

    const byteLength = this.byteLength;

    if (src instanceof GPUBuffer) {
      if (srcOffset % 4 !== 0) {
        throw new Error(`srcOffset (${srcOffset}) must be a multiple of 4 bytes`);
      }
      if (src.size < srcOffset + byteLength) {
        throw new Error(`GPUBuffer size (${src.size}B) < required (${srcOffset + byteLength}B)`);
      }
      if ((src.usage & GPUBufferUsage.COPY_SRC) === 0) {
        throw new Error('GPUBuffer must have GPUBufferUsage.COPY_SRC flag');
      }

      const encoder = this.#device.createCommandEncoder();
      encoder.copyBufferToBuffer(src, srcOffset, this.#buffer, this.#byteOffset, byteLength);
      this.#device.queue.submit([encoder.finish()]);
      return this;
    }

    if (src.byteLength < srcOffset + byteLength) {
      throw new Error(`Source bytes (${src.byteLength}B) < required (${srcOffset + byteLength}B)`);
    }

    this.#device.queue.writeBuffer(
      this.#buffer,
      this.#byteOffset,
      src.buffer,
      src.byteOffset + srcOffset,
      byteLength
    );
    return this;
  }

  /**
   * Reads raw bytes back from the GPU storage buffer slice into host memory.
   *
   * @returns Raw byte buffer containing the tensor slice data.
   */
  async getData(): Promise<ArrayBuffer> {
    const byteLength = this.byteLength;
    const alignedSize = Math.max(16, Math.ceil(byteLength / 4) * 4);

    const encoder = this.#device.createCommandEncoder();
    const staging = this.#device.createBuffer({ size: alignedSize, usage: STAGING_BUFFER_USAGE });

    try {
      encoder.copyBufferToBuffer(this.#buffer, this.#byteOffset, staging, 0, alignedSize);
      this.#device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      return staging.getMappedRange(0, byteLength).slice(0);
    } finally {
      staging.destroy();
    }
  }
}
