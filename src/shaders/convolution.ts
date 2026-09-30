import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

const name = 'aten.convolution.default';

/**
 * WGSL compute shader for direct 2D convolution (non-transposed).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/et_vk_conv2d/conv2d.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_CONV2D = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  B: u32,
  IC: u32,
  IH: u32,
  IW: u32,
  OC: u32,
  OH: u32,
  OW: u32,
  KH: u32,
  KW: u32,
  sH: u32,
  sW: u32,
  pH: u32,
  pW: u32,
  dH: u32,
  dW: u32,
  groups: u32,
  has_bias: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 256;

// Direct 2D convolution (non-transposed), NCHW row-major, fp32. ONE thread per
// (b, oc, oh, ow) output element. Supports general stride/padding/dilation and
// groups. input [B,IC,IH,IW]; weight [OC, IC/groups, KH, KW]; bias [OC] (gated).
@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let total = params.B * params.OC * params.OH * params.OW;
  let i = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let oc = (i / (params.OW * params.OH)) % params.OC;
  let b = i / (params.OW * params.OH * params.OC);

  let icpg = params.IC / params.groups; // input channels per group
  let ocpg = params.OC / params.groups; // output channels per group
  let g = oc / ocpg;
  let ic0 = g * icpg;

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = bias[oc];
  }

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  for (var icg: u32 = 0u; icg < icpg; icg = icg + 1u) {
    let ic = ic0 + icg;
    let in_c_base = (b * params.IC + ic) * params.IH; // *IW added per-row below
    let w_c_base = (oc * icpg + icg) * params.KH; // *KW added per-row below
    for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
      let ih = i32(oh) * i32(params.sH) - i32(params.pH) + i32(kh) * i32(params.dH);
      if (ih < 0 || ih >= iH) {
        continue;
      }
      let in_row = (in_c_base + u32(ih)) * params.IW;
      let w_row = (w_c_base + kh) * params.KW;
      for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
        let iw = i32(ow) * i32(params.sW) - i32(params.pW) + i32(kw) * i32(params.dW);
        if (iw < 0 || iw >= iW) {
          continue;
        }
        acc = acc + input[in_row + u32(iw)] * weight[w_row + kw];
      }
    }
  }
  out[i] = acc;
}
`;

/**
 * WGSL compute shader for transposed 2D convolution (gather form).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/et_vk_conv2d/conv_transpose2d.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/et_vk_conv2d/conv_transpose2d.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_CONV_TRANSPOSE2D = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  B: u32,
  IC: u32,
  IH: u32,
  IW: u32,
  OC: u32,
  OH: u32,
  OW: u32,
  KH: u32,
  KW: u32,
  sH: u32,
  sW: u32,
  pH: u32,
  pW: u32,
  dH: u32,
  dW: u32,
  groups: u32,
  has_bias: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 256;

// Transposed 2D convolution (gather form), NCHW row-major, fp32. ONE thread per
// (b, oc, oh, ow) output element. weight layout = torch convT [IC, OC/groups,
// KH, KW] (NOT flipped). For each kernel tap (kh,kw): an input row ih
// contributes iff (oh + pH - kh*dH) is divisible by sH and ih in range (the
// scatter-inversion). CPU-derisked vs torch.conv_transpose2d to fp64 round-off
// (/tmp/convtr_derisk.py), incl. non-square spatial + non-square kernel.
@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let total = params.B * params.OC * params.OH * params.OW;
  let i = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let oc = (i / (params.OW * params.OH)) % params.OC;
  let b = i / (params.OW * params.OH * params.OC);

  let icpg = params.IC / params.groups; // input channels per group
  let ocpg = params.OC / params.groups; // output channels per group
  let g = oc / ocpg;
  let ic0 = g * icpg;
  let oc_in_g = oc % ocpg;

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = bias[oc];
  }

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
    let num_h = i32(oh) + i32(params.pH) - i32(kh) * i32(params.dH);
    if (num_h % i32(params.sH) != 0) {
      continue;
    }
    let ih = num_h / i32(params.sH);
    if (ih < 0 || ih >= iH) {
      continue;
    }
    for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
      let num_w = i32(ow) + i32(params.pW) - i32(kw) * i32(params.dW);
      if (num_w % i32(params.sW) != 0) {
        continue;
      }
      let iw = num_w / i32(params.sW);
      if (iw < 0 || iw >= iW) {
        continue;
      }
      for (var icg: u32 = 0u; icg < icpg; icg = icg + 1u) {
        let ic = ic0 + icg;
        let in_idx =
            ((b * params.IC + ic) * params.IH + u32(ih)) * params.IW + u32(iw);
        // weight [IC, OC/groups, KH, KW]: index (ic, oc_in_g, kh, kw)
        let w_idx =
            ((ic * ocpg + oc_in_g) * params.KH + kh) * params.KW + kw;
        acc = acc + input[in_idx] * weight[w_idx];
      }
    }
  }
  out[i] = acc;
}
`;

/**
 * Positional argument tuple for `aten::convolution.default`:
 * `[in, weight, bias, stride, padding, dilation, transposed, output_padding, groups, out]`
 */
export type ConvolutionArgs = readonly [
  inTensor: Tensor,
  weight: Tensor,
  bias: Tensor | undefined,
  stride: readonly number[],
  padding: readonly number[],
  dilation: readonly number[],
  transposed: boolean,
  outputPadding: readonly number[],
  groups: number,
  outTensor: Tensor,
];

function parseHw(v: readonly number[], argName: string): [number, number] {
  if (v.length === 1) return [v[0], v[0]];
  if (v.length === 2) return [v[0], v[1]];
  throw new Error(`${name}: ${argName} must be 1 or 2 elements`);
}

/**
 * Internal dispatch builder for `aten::convolution.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: ConvolutionArgs): void {
  const [
    inTensor,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    outputPadding,
    groups,
    outTensor,
  ] = args;

  if (inTensor.dtype !== 'float32' || weight.dtype !== 'float32' || outTensor.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 tensors are currently supported`);
  }
  if (inTensor.shape.length !== 4 || weight.shape.length !== 4 || outTensor.shape.length !== 4) {
    throw new Error(`${name}: expected 4D tensors for input, weight, and output`);
  }

  const [sH, sW] = parseHw(stride, 'stride');
  const [pH, pW] = parseHw(padding, 'padding');
  const [dH, dW] = parseHw(dilation, 'dilation');
  const [opH, opW] = parseHw(outputPadding, 'output_padding');

  const [b, ic, ih, iw] = inTensor.shape as [number, number, number, number];
  const hasBias = bias !== undefined;

  let oc: number;
  let kh: number;
  let kw: number;

  if (transposed) {
    // weight layout for transposed conv: [IC, OC / groups, KH, KW]
    kh = weight.shape[2];
    kw = weight.shape[3];
    const ocpg = weight.shape[1];
    oc = ocpg * groups;

    if (weight.shape[0] !== ic) {
      throw new Error(`${name}: weight dim0 !== IC (${weight.shape[0]} != ${ic})`);
    }
    if (opH >= sH || opW >= sW) {
      throw new Error(`${name}: output_padding >= stride`);
    }

    const expectedOh = (ih - 1) * sH - 2 * pH + dH * (kh - 1) + opH + 1;
    const expectedOw = (iw - 1) * sW - 2 * pW + dW * (kw - 1) + opW + 1;
    if (outTensor.shape[2] !== expectedOh || outTensor.shape[3] !== expectedOw) {
      throw new Error(`${name}: expected output spatial [${expectedOh}, ${expectedOw}]`);
    }
  } else {
    // weight layout for direct conv: [OC, IC / groups, KH, KW]
    oc = weight.shape[0];
    kh = weight.shape[2];
    kw = weight.shape[3];

    if (outputPadding.some((v) => v !== 0)) {
      throw new Error(`${name}: non-zero output_padding unsupported for non-transposed conv`);
    }
    if (weight.shape[1] !== ic / groups) {
      throw new Error(`${name}: weight in-ch (${weight.shape[1]}) != IC/groups (${ic / groups})`);
    }

    const ohNum = ih + 2 * pH - dH * (kh - 1) - 1;
    const owNum = iw + 2 * pW - dW * (kw - 1) - 1;
    if (ohNum < 0 || owNum < 0) {
      throw new Error(`${name}: invalid geometry (kernel larger than input)`);
    }
    const expectedOh = Math.floor(ohNum / sH) + 1;
    const expectedOw = Math.floor(owNum / sW) + 1;
    if (outTensor.shape[2] !== expectedOh || outTensor.shape[3] !== expectedOw) {
      throw new Error(`${name}: expected output spatial [${expectedOh}, ${expectedOw}]`);
    }
  }

  const [, , oh, ow] = outTensor.shape as [number, number, number, number];

  const device = ctx.device;
  const wgSize = 256;
  const totalWorkgroups = Math.ceil(outTensor.numel / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

  const buffer = new ArrayBuffer(80);
  new Uint32Array(buffer).set([
    b,
    ic,
    ih,
    iw,
    oc,
    oh,
    ow,
    kh,
    kw,
    sH,
    sW,
    pH,
    pW,
    dH,
    dW,
    groups,
    hasBias ? 1 : 0,
  ]);

  const uniformBuffer = ctx.uniformBuffer(buffer);
  const biasBuffer = hasBias ? bias.buffer : weight.buffer;

  const code = transposed ? SHADER_CONV_TRANSPOSE2D : SHADER_CONV2D;
  const bundle = createComputeBundle(
    device,
    code,
    [
      { binding: 0, buffer: outTensor.buffer },
      { binding: 1, buffer: inTensor.buffer },
      { binding: 2, buffer: weight.buffer },
      { binding: 3, buffer: biasBuffer },
      { binding: 4, buffer: uniformBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize }
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::convolution.default`.
 *
 * Implements 2D dense and grouped convolutions as well as 2D transposed convolutions.
 */
export const convolution: Shader<ConvolutionArgs, typeof name> = {
  name,
  code: { conv2d: SHADER_CONV2D, convTranspose2d: SHADER_CONV_TRANSPOSE2D },
  recordIn,
};
