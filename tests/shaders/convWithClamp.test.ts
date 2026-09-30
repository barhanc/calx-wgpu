import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('et_vk.conv_with_clamp.default', () => {
  it('computes 2D dense convolution with clamp and bias', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // input: [1, 1, 3, 3]
    const inp = ctx
      .tensor('float32', [1, 1, 3, 3])
      .setData(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));

    // weight: [1, 1, 2, 2]
    const weight = ctx.tensor('float32', [1, 1, 2, 2]).setData(new Float32Array([1, 0, 0, 1]));

    // bias: [1]
    const bias = ctx.tensor('float32', [1]).setData(new Float32Array([10]));

    // out: [1, 1, 2, 2] with stride 1, pad 0, dilation 1
    // Unclamped conv results: [16, 18, 22, 24]
    // Clamped with min: 17.0, max: 23.0 -> [17, 18, 22, 23]
    const out = ctx.tensor('float32', [1, 1, 2, 2]);

    ctx.recordShader(shaders.convWithClamp, [
      inp,
      weight,
      bias,
      [1, 1], // stride
      [0, 0], // padding
      [1, 1], // dilation
      false, // transposed
      [0, 0], // output_padding
      1, // groups
      17.0, // min
      23.0, // max
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [17, 18, 22, 23]);
    ctx.destroy();
  });

  it('computes depthwise grouped convolution with clamp (ReLU6 style)', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // input: [1, 2, 2, 2]
    const inp = ctx.tensor('float32', [1, 2, 2, 2]).setData(
      new Float32Array([
        // channel 0
        -5, -2, 1, 2,
        // channel 1
        2, 4, 1, 3,
      ])
    );

    // weight: [2, 1, 2, 2] with groups = 2 (depthwise)
    const weight = ctx.tensor('float32', [2, 1, 2, 2]).setData(
      new Float32Array([
        // filter 0 for channel 0: sum -> -5 + -2 + 1 + 2 = -4
        1, 1, 1, 1,
        // filter 1 for channel 1: sum -> 2 + 4 + 1 + 3 = 10
        1, 1, 1, 1,
      ])
    );

    const out = ctx.tensor('float32', [1, 2, 1, 1]);

    // clamp [0.0, 6.0]
    // ch0: clamp(-4, 0, 6) = 0
    // ch1: clamp(10, 0, 6) = 6
    ctx.recordShader(shaders.convWithClamp, [
      inp,
      weight,
      undefined, // no bias
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      2, // groups
      0.0,
      6.0,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [0, 6]);
    ctx.destroy();
  });
});
