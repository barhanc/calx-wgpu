declare const tensorBrand: unique symbol;

/**
 * Supported tensor data types in WebGPU storage buffers.
 */
export type DType = 'float32' | 'float16' | 'int32' | 'int8' | 'uint8' | 'bool';

/**
 * A WebGPU-backed Tensor instance.
 *
 * Encapsulates an allocated `GPUBuffer` in device VRAM along with its metadata
 * (`dtype`, `shape`, `numel`). Use {@link Tensor.getData} to read raw bytes
 * back into host memory, or {@link Tensor.destroy} to release GPU memory.
 */
export type Tensor = {
  /** The element data type of the tensor. */
  readonly dtype: DType;
  /** Total number of elements across all dimensions. */
  readonly numel: number;
  /** The dimensions of the tensor. */
  readonly shape: readonly number[];
  /** The WebGPU device that owns this tensor's buffer. */
  readonly device: GPUDevice;
  /** The underlying WebGPU storage buffer in VRAM. */
  readonly buffer: GPUBuffer;

  /**
   * Reads raw bytes back from the GPU storage buffer into host memory.
   * @returns Raw byte buffer containing the tensor data.
   */
  readonly getData: () => Promise<ArrayBuffer>;

  /** Destroys and releases the underlying GPU buffer. */
  readonly destroy: () => void;

  /**
   * Prevents plain JS objects from being cast as Tensors. Tensors should only
   * be created via the `tensor` function exported from this module.
   * @internal
   */
  readonly [tensorBrand]: never;
};

// prettier-ignore
const DTYPE_BYTESIZE: Record<DType, number> = {
  float32: 4, float16: 2,
  int32:   4, int8:    1,
  uint8:   1, bool:    1,
} as const;

// prettier-ignore
const STORAGE_BUFFER_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const STAGING_BUFFER_USAGE = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;

/**
 * Maximum tensor rank supported by WebGPU kernels (std140 layout limit).
 */
export const MAX_NDIM = 8;

/**
 * Creates a WebGPU-backed Tensor.
 *
 * If `source` is omitted, allocates an uninitialized storage buffer in device VRAM.
 * If `source` is a `TypedArray`, uploads data to a newly allocated GPU storage buffer.
 * If `source` is an existing `GPUBuffer`, wraps it zero-copy without allocating new memory.
 *
 * @param dtype Data type of elements.
 * @param shape Dimensions of the tensor (rank <= 8).
 * @param device The WebGPU device.
 * @param source Optional initial host data or existing GPUBuffer to wrap.
 * @returns A newly created Tensor closure bundle.
 */
export function tensor(
  dtype: DType,
  shape: readonly number[],
  device: GPUDevice,
  source?: GPUBuffer | Float32Array | Int32Array | Int8Array | Uint8Array | Uint16Array
): Tensor {
  if (shape.length > MAX_NDIM) {
    throw new Error(`Tensor rank ${shape.length} exceeds maximum rank of ${MAX_NDIM}`);
  }
  if (shape.some((dim) => dim <= 0 || !Number.isInteger(dim))) {
    throw new Error('Tensor dimensions must be positive integers');
  }

  const numel = shape.reduce((a, b) => a * b, 1);
  const byteLength = numel * DTYPE_BYTESIZE[dtype];
  const alignedSize = Math.max(16, Math.ceil(byteLength / 4) * 4);

  let buffer: GPUBuffer;
  if (!source) {
    buffer = device.createBuffer({ size: alignedSize, usage: STORAGE_BUFFER_USAGE });
  } else if (source instanceof GPUBuffer) {
    if (source.size < byteLength) {
      throw new Error(`GPUBuffer size (${source.size}B) < required tensor size (${byteLength}B)`);
    }
    if ((source.usage & GPUBufferUsage.STORAGE) === 0) {
      throw new Error('GPUBuffer must have GPUBufferUsage.STORAGE flag');
    }
    buffer = source;
  } else {
    if (source.byteLength !== byteLength) {
      throw new Error(`Source bytes (${source.byteLength}B) !== tensor size (${byteLength}B)`);
    }
    buffer = device.createBuffer({ size: alignedSize, usage: STORAGE_BUFFER_USAGE });
    device.queue.writeBuffer(buffer, 0, source.buffer, source.byteOffset, byteLength);
  }

  const attributes = { dtype, shape, numel, device, buffer };

  const destroy = () => buffer.destroy();

  const getData = async (): Promise<ArrayBuffer> => {
    const encoder = device.createCommandEncoder();
    const staging = device.createBuffer({ size: alignedSize, usage: STAGING_BUFFER_USAGE });

    try {
      encoder.copyBufferToBuffer(buffer, 0, staging, 0, alignedSize);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      return staging.getMappedRange(0, byteLength).slice(0);
    } finally {
      staging.destroy();
    }
  };

  return { ...attributes, getData, destroy } as Tensor;
}
