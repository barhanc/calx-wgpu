import { add } from './add';
import { constantPadNd } from './constant-pad-nd';
import { convolution } from './convolution';
import { convWithClamp } from './conv-with-clamp';
import { hardswish } from './hardswish';
import { meanDim } from './mean-dim';
import { mm } from './mm';
import { mul } from './mul';
import { sigmoid } from './sigmoid';
import { upsampleBilinear2d } from './upsample-bilinear2d';

/**
 * Namespace containing all registered WebGPU compute shaders.
 */
// prettier-ignore
export const shaders = {
  add, constantPadNd, convolution, convWithClamp, hardswish, meanDim, mm, mul, sigmoid, upsampleBilinear2d,
} as const;

/**
 * Maps ExecuTorch operator names to their shader implementations.
 *
 * @internal
 */
export const shaderRegistry = Object.fromEntries(
  Object.values(shaders).map((shader) => [shader.name, shader])
);
