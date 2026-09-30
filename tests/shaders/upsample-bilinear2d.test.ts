import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.upsample_bilinear2d.vec', () => {
  it('upsamples 2x2 to 4x4 with align_corners=false', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // input: [1, 1, 2, 2] -> [[1.0, 2.0], [3.0, 4.0]]
    const inp = ctx.tensor('float32', [1, 1, 2, 2]).setData(new Float32Array([1.0, 2.0, 3.0, 4.0]));
    const out = ctx.tensor('float32', [1, 1, 4, 4]);

    ctx.recordShader(shaders.upsampleBilinear2d, [
      inp,
      [4, 4], // output_size
      false, // align_corners
      undefined, // scales
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    const expected = [
      1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0,
    ];
    await expectTensorClose(out, expected);
    ctx.destroy();
  });

  it('upsamples 2x2 to 4x4 with align_corners=true', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float32', [1, 1, 2, 2]).setData(new Float32Array([1.0, 2.0, 3.0, 4.0]));
    const out = ctx.tensor('float32', [1, 1, 4, 4]);

    ctx.recordShader(shaders.upsampleBilinear2d, [inp, [4, 4], true, undefined, out]);
    ctx.submit();
    await ctx.sync();

    const expected = [
      1.0,
      1.0 + 1.0 / 3.0,
      1.0 + 2.0 / 3.0,
      2.0,
      1.0 + 2.0 / 3.0,
      2.0,
      2.0 + 1.0 / 3.0,
      2.0 + 2.0 / 3.0,
      2.0 + 1.0 / 3.0,
      2.0 + 2.0 / 3.0,
      3.0,
      3.0 + 1.0 / 3.0,
      3.0,
      3.0 + 1.0 / 3.0,
      3.0 + 2.0 / 3.0,
      4.0,
    ];
    await expectTensorClose(out, expected);
    ctx.destroy();
  });
});
