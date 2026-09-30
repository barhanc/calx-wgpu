import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.mean.dim', () => {
  it('computes 2D spatial mean reduction (global average pooling)', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // input: [1, 2, 2, 2]
    // ch 0: [[1, 2], [3, 4]] -> mean = 2.5
    // ch 1: [[10, 20], [30, 40]] -> mean = 25.0
    const inp = ctx.tensor('float32', [1, 2, 2, 2]).setData(
      new Float32Array([
        // channel 0
        1, 2, 3, 4,
        // channel 1
        10, 20, 30, 40,
      ])
    );

    const out = ctx.tensor('float32', [1, 2, 1, 1]);

    ctx.recordShader(shaders.meanDim, [
      inp,
      [2, 3], // reduce over H and W
      true, // keepdim
      undefined, // dtype
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [2.5, 25.0]);
    ctx.destroy();
  });

  it('computes single dimension reduction across columns', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // input: [2, 3]
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // mean across dim 1 (columns) -> [2, 5]
    const inp = ctx.tensor('float32', [2, 3]).setData(new Float32Array([1, 2, 3, 4, 5, 6]));

    const out = ctx.tensor('float32', [2, 1]);

    ctx.recordShader(shaders.meanDim, [inp, [1], true, undefined, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [2, 5]);
    ctx.destroy();
  });

  it('computes single dimension reduction across rows', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // input: [2, 3]
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // mean across dim 0 (rows) -> [2.5, 3.5, 4.5]
    const inp = ctx.tensor('float32', [2, 3]).setData(new Float32Array([1, 2, 3, 4, 5, 6]));

    const out = ctx.tensor('float32', [1, 3]);

    ctx.recordShader(shaders.meanDim, [inp, [0], true, undefined, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [2.5, 3.5, 4.5]);
    ctx.destroy();
  });
});
