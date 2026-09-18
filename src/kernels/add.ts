import type { Kernel } from '../core/kernel';
import type { WgpuExecutionContext } from '../core/context';
import { compute2DWorkgroupCount, createComputeBundle } from '../core/dispatch';

import { isBroadcastable } from './utils/broadcast';
import { createTensorMetaBuffer, TENSOR_META_WGSL } from './utils/meta';

const NAME = 'aten::add.Tensor';

/**
 * WGSL compute shader for binary addition with broadcast support,
 * copied directly from ExecuTorch backends/webgpu/runtime/ops/add/binary_add.wgsl.
 */
const SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

${TENSOR_META_WGSL}
@group(0) @binding(3) var<uniform> out_meta: TensorMeta;
@group(0) @binding(4) var<uniform> in1_meta: TensorMeta;
@group(0) @binding(5) var<uniform> in2_meta: TensorMeta;

override wg_size: u32 = 256u;
override alpha: f32 = 1.0;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= out_meta.numel) {
    return;
  }

  var same = true;
  for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
    if (
      in1_meta.sizes[d >> 2u][d & 3u] != out_meta.sizes[d >> 2u][d & 3u] ||
      in2_meta.sizes[d >> 2u][d & 3u] != out_meta.sizes[d >> 2u][d & 3u]
    ) {
      same = false;
    }
  }
  if (same) {
    output[idx] = input1[idx] + alpha * input2[idx];
    return;
  }

  var rem = idx;
  var l1: u32 = 0u;
  var l2: u32 = 0u;
  for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
    let coord = rem / out_meta.strides[d >> 2u][d & 3u];
    rem = rem % out_meta.strides[d >> 2u][d & 3u];
    l1 = l1 + min(coord, in1_meta.sizes[d >> 2u][d & 3u] - 1u) * in1_meta.strides[d >> 2u][d & 3u];
    l2 = l2 + min(coord, in2_meta.sizes[d >> 2u][d & 3u] - 1u) * in2_meta.strides[d >> 2u][d & 3u];
  }
  output[idx] = input1[l1] + alpha * input2[l2];
}
`;

/**
 * Argument schema for `aten::add.Tensor`.
 */
export type AddArgs = {
  /** First input tensor value ID in the execution context. */
  readonly in1: number;
  /** Second input tensor value ID in the execution context. */
  readonly in2: number;
  /** Output tensor value ID in the execution context. */
  readonly out: number;
  /** Optional alpha scalar constant value ID in the execution context (defaults to 1.0). */
  readonly alpha?: number;
};

/**
 * Internal dispatch builder for `aten::add.Tensor`.
 * @param ctx The execution context.
 * @param args Named argument value IDs for the operator.
 */
function attachTo(ctx: WgpuExecutionContext, args: AddArgs): void {
  const in1 = ctx.getTensor(args.in1);
  const in2 = ctx.getTensor(args.in2);
  const out = ctx.getTensor(args.out);
  const alpha = args.alpha !== undefined ? ctx.getScalar(args.alpha) : 1.0;

  if (in1.dtype !== 'float32' || in2.dtype !== 'float32' || out.dtype !== 'float32') {
    throw new Error(`${NAME}: Only float32 tensors are currently supported`);
  }
  if (!isBroadcastable(in1.shape, out.shape) || !isBroadcastable(in2.shape, out.shape)) {
    throw new Error(`${NAME}: Input shapes are not broadcastable to output shape`);
  }

  const outRank = out.shape.length;
  const outMetaBuffer = createTensorMetaBuffer(out);
  const in1MetaBuffer = createTensorMetaBuffer(in1, outRank);
  const in2MetaBuffer = createTensorMetaBuffer(in2, outRank);

  const device = ctx.device;
  const wgSize = 256;
  const { workgroupCountX, workgroupCountY } = compute2DWorkgroupCount(out.numel, wgSize);

  const bundle = createComputeBundle(
    device,
    SHADER,
    [
      { binding: 0, buffer: in1.buffer },
      { binding: 1, buffer: in2.buffer },
      { binding: 2, buffer: out.buffer },
      { binding: 3, buffer: outMetaBuffer },
      { binding: 4, buffer: in1MetaBuffer },
      { binding: 5, buffer: in2MetaBuffer },
    ],
    // eslint-disable-next-line camelcase
    { wg_size: wgSize, alpha }
  );

  ctx.ownBuffer(outMetaBuffer);
  ctx.ownBuffer(in1MetaBuffer);
  ctx.ownBuffer(in2MetaBuffer);

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::add.Tensor`.
 *
 * Computes elementwise `output = input1 + alpha * input2` with NumPy/PyTorch-style
 * broadcasting (right-aligned, dimensions equal or 1, up to rank 8). Features an
 * automatic fast path when input and output shapes match.
 *
 * @example
 * ```ts
 * kernels.add.attachTo(ctx, { in1: 0, in2: 1, out: 3, alpha: 2 });
 * ```
 */
export const add: Kernel<AddArgs> = {
  name: NAME,
  wgsl: SHADER,
  attachTo,
};
