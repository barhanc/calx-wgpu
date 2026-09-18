/**
 * Core Tensor type and factory for WebGPU execution.
 * Pure functional closure bundle over a WebGPU storage buffer.
 */

/**
 * Supported data types in WebGPU storage buffers.
 * @category Core / Types
 */
export type DType = 'float32' | 'float16' | 'int32' | 'int8' | 'uint8' | 'bool';

/**
 * JavaScript TypedArrays supported for CPU host data transfers.
 * @category Core / Types
 */
export type TypedArray = Float32Array | Int32Array | Int8Array | Uint8Array | Uint16Array;

/**
 * Byte sizes for each supported data type.
 * @category Core / Constants
 */
export const DTYPE_BYTE_SIZES: Record<DType, number> = {
  float32: 4,
  float16: 2,
  int32: 4,
  int8: 1,
  uint8: 1,
  bool: 1,
};

/**
 * A WebGPU-backed Tensor instance.
 * @category Core / Types
 */
export type Tensor = {
  readonly dtype: DType;
  readonly shape: readonly number[];
  readonly strides: readonly number[];
  readonly numel: number;
  readonly byteLength: number;
  readonly buffer: GPUBuffer;
  readonly byteOffset: number;
  readonly device: GPUDevice;

  /** Reads GPU data back to CPU into a TypedArray */
  readonly toArray: () => Promise<TypedArray>;

  /** Writes CPU data into the GPU buffer */
  readonly setData: (src: ArrayBufferView | ArrayBuffer) => void;

  /** Fast GPU-to-GPU buffer copy (no CPU roundtrip) */
  readonly copyTo: (dst: Tensor, options?: { offset?: number; length?: number }) => void;

  /** Releases the underlying GPU buffer */
  readonly dispose: () => void;
};

/**
 * Configuration options for creating a Tensor.
 * @category Core / Types
 */
export type TensorOptions = {
  readonly byteOffset?: number;
  readonly buffer?: GPUBuffer;
  readonly destroyOnDispose?: boolean;
};

/**
 * Computes contiguous row-major strides from shape.
 * @param shape Tensor dimensions.
 * @returns Strides array.
 */
export const computeStrides = (shape: readonly number[]): readonly number[] => {
  const rank = shape.length;
  const strides = new Array<number>(rank);
  let acc = 1;
  for (let i = rank - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i];
  }
  return strides;
};

/**
 * Computes total number of elements from shape dimensions.
 * @param shape Tensor dimensions.
 * @returns Total number of elements.
 */
export const computeNumel = (shape: readonly number[]): number => {
  return shape.reduce((acc, dim) => acc * dim, 1);
};

/**
 * Aligns byte size to WebGPU 4-byte buffer alignment.
 * @param size Raw byte size.
 * @returns 4-byte aligned size.
 */
export const alignTo4 = (size: number): number => {
  return Math.ceil(size / 4) * 4;
};

/**
 * Creates a Tensor instance backed by a WebGPU storage buffer.
 *
 * @param dtype Data type ('float32', 'int32', etc.).
 * @param shape Tensor dimensions (e.g. [1, 3, 224, 224]).
 * @param src Optional initial data (TypedArray, ArrayBuffer, or existing GPUBuffer).
 * @param device WebGPU device.
 * @param options Optional buffer configuration.
 * @returns A newly created Tensor closure bundle.
 */
export const tensor = (
  dtype: DType,
  shape: readonly number[],
  src?: ArrayBufferView | ArrayBuffer | GPUBuffer,
  device?: GPUDevice,
  options?: TensorOptions
): Tensor => {
  const rank = shape.length;
  if (rank > 8) {
    throw new Error(`Tensor rank ${rank} exceeds maximum supported rank of 8`);
  }

  for (const d of shape) {
    if (d <= 0 || !Number.isInteger(d)) {
      throw new Error(`Invalid tensor dimension ${d}: dimensions must be positive integers`);
    }
  }

  const strides = computeStrides(shape);
  const numel = computeNumel(shape);
  const elemSize = DTYPE_BYTE_SIZES[dtype];
  const byteLength = numel * elemSize;
  const byteOffset = options?.byteOffset ?? 0;
  const alignedSize = Math.max(16, alignTo4(byteLength));

  let gpuBuffer: GPUBuffer;
  let dev: GPUDevice;
  let shouldDestroy = options?.destroyOnDispose ?? true;

  if (src instanceof GPUBuffer) {
    gpuBuffer = src;
    if (!device) {
      throw new Error('GPUDevice must be provided when creating a Tensor from a GPUBuffer');
    }
    dev = device;
    if (options?.destroyOnDispose === undefined) {
      shouldDestroy = false;
    }
  } else {
    if (!device) {
      throw new Error('GPUDevice is required to allocate a Tensor buffer');
    }
    dev = device;
    gpuBuffer =
      options?.buffer ??
      dev.createBuffer({
        size: alignedSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

    if (src) {
      if (ArrayBuffer.isView(src)) {
        dev.queue.writeBuffer(
          gpuBuffer,
          byteOffset,
          src.buffer,
          src.byteOffset,
          Math.min(src.byteLength, byteLength)
        );
      } else {
        dev.queue.writeBuffer(gpuBuffer, byteOffset, src, 0, Math.min(src.byteLength, byteLength));
      }
    }
  }

  let disposed = false;

  const toArray = async (): Promise<TypedArray> => {
    if (disposed) {
      throw new Error('Cannot read from a disposed Tensor');
    }

    const stagingBuffer = dev.createBuffer({
      size: alignedSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const commandEncoder = dev.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(gpuBuffer, byteOffset, stagingBuffer, 0, alignedSize);
    dev.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const copy = stagingBuffer.getMappedRange(0, byteLength).slice(0);
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    switch (dtype) {
      case 'float32':
        return new Float32Array(copy);
      case 'int32':
        return new Int32Array(copy);
      case 'int8':
        return new Int8Array(copy);
      case 'uint8':
      case 'bool':
        return new Uint8Array(copy);
      case 'float16':
        return new Uint16Array(copy);
    }
  };

  const setData = (newData: ArrayBufferView | ArrayBuffer): void => {
    if (disposed) {
      throw new Error('Cannot write to a disposed Tensor');
    }

    if (ArrayBuffer.isView(newData)) {
      dev.queue.writeBuffer(
        gpuBuffer,
        byteOffset,
        newData.buffer,
        newData.byteOffset,
        Math.min(newData.byteLength, byteLength)
      );
    } else {
      dev.queue.writeBuffer(
        gpuBuffer,
        byteOffset,
        newData,
        0,
        Math.min(newData.byteLength, byteLength)
      );
    }
  };

  const copyTo = (dst: Tensor, copyOptions?: { offset?: number; length?: number }): void => {
    if (disposed) {
      throw new Error('Cannot copy from a disposed Tensor');
    }
    if (dst.dtype !== dtype) {
      throw new Error(`Data type mismatch in copyTo: source is ${dtype}, dest is ${dst.dtype}`);
    }

    const elemOffset = copyOptions?.offset ?? 0;
    const elemLength = copyOptions?.length ?? numel - elemOffset;

    const copyBytes = elemLength * elemSize;
    const srcByteStart = byteOffset + elemOffset * elemSize;
    const dstByteStart = dst.byteOffset;

    const commandEncoder = dev.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      gpuBuffer,
      srcByteStart,
      dst.buffer,
      dstByteStart,
      alignTo4(copyBytes)
    );
    dev.queue.submit([commandEncoder.finish()]);
  };

  const dispose = (): void => {
    if (!disposed) {
      disposed = true;
      if (shouldDestroy) {
        gpuBuffer.destroy();
      }
    }
  };

  return {
    dtype,
    shape,
    strides,
    numel,
    byteLength,
    buffer: gpuBuffer,
    byteOffset,
    device: dev,
    toArray,
    setData,
    copyTo,
    dispose,
  };
};
