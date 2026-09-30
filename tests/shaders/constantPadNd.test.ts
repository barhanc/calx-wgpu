import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.constant_pad_nd.default', () => {
  it('pads 4D tensor with zeros (pad last two dims)', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // [1, 1, 2, 2] -> pad width [left 1, right 1], height [top 0, bottom 1] -> [1, 1, 3, 4]
    // pad: [1, 1, 0, 1] (reversed dim order: W left, W right, H top, H bottom)
    const inp = ctx.tensor('float32', [1, 1, 2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const out = ctx.tensor('float32', [1, 1, 3, 4]);

    ctx.recordShader(shaders.constantPadNd, [inp, [1, 1, 0, 1], 0.0, out]);
    ctx.submit();
    await ctx.sync();

    // Expected shape [1, 1, 3, 4]:
    // Row 0 (top=0): [0, 1, 2, 0]
    // Row 1:         [0, 3, 4, 0]
    // Row 2 (bottom=1): [0, 0, 0, 0]
    await expectTensorClose(out, [0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0]);
    ctx.destroy();
  });

  it('pads with custom fill value', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // [2, 2] -> pad [1, 0, 0, 1] (W left 1, W right 0, H top 0, H bottom 1) -> [3, 3]
    const inp = ctx.tensor('float32', [2, 2]).setData(new Float32Array([5, 6, 7, 8]));
    const out = ctx.tensor('float32', [3, 3]);

    ctx.recordShader(shaders.constantPadNd, [inp, [1, 0, 0, 1], -1.0, out]);
    ctx.submit();
    await ctx.sync();

    // Row 0: [-1, 5, 6]
    // Row 1: [-1, 7, 8]
    // Row 2: [-1, -1, -1]
    await expectTensorClose(out, [-1, 5, 6, -1, 7, 8, -1, -1, -1]);
    ctx.destroy();
  });

  it('runs inside a recordProgram sequence', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const program = {
      version: '1',
      chain: [{ name: 'aten.constant_pad_nd.default', args: [0, 1, 2, 3] }],
      values: [
        { type: 'tensor', shape: [1, 2], dtype: 'float32', memObjId: 0 },
        { type: 'list', items: [1, 1] },
        { type: 'scalar', value: 9.0 },
        { type: 'tensor', shape: [1, 4], dtype: 'float32', memObjId: 1 },
      ],
      inputIds: [0],
      outputIds: [3],
      memoryPlan: {
        pools: [
          { id: 0, size: 8 },
          { id: 1, size: 16 },
        ],
      },
    } as const;

    const inp = ctx.tensor('float32', [1, 2]).setData(new Float32Array([10, 20]));
    const out = ctx.tensor('float32', [1, 4]);

    ctx.recordProgram(program, new ArrayBuffer(0), [inp, out]);
    ctx.submit();
    await ctx.sync();

    // pad [1, 1] on 1D/last dim -> [9, 10, 20, 9]
    await expectTensorClose(out, [9, 10, 20, 9]);
    ctx.destroy();
  });
});
