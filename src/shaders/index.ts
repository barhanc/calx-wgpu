import { add } from './add';
import { constantPadNd } from './constantPadNd';
import { convolution } from './convolution';
import { convWithClamp } from './convWithClamp';
import { hardswish } from './hardswish';
import { meanDim } from './meanDim';
import { mm } from './mm';
import { mul } from './mul';
import { sigmoid } from './sigmoid';

/**
 * Namespace containing all registered WebGPU compute shaders.
 */
// prettier-ignore
export const shaders = {
  add, constantPadNd, convolution, convWithClamp, hardswish, meanDim, mm, mul, sigmoid,
} as const;

/**
 * Maps ExecuTorch operator names to their shader implementations.
 *
 * @internal
 */
export const shaderRegistry = Object.fromEntries(
  Object.values(shaders).map((shader) => [shader.name, shader])
);
