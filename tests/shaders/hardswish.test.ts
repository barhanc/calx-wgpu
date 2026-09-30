import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.hardswish.default', () => {
  it('computes hardswish activation values across regions', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // Regions:
    // x <= -3.0 -> 0
    // -3.0 < x < 3.0 -> x * (x + 3.0) / 6.0
    // x >= 3.0 -> x
    const inputs = [-5.0, -3.0, -1.5, 0.0, 1.0, 2.0, 3.0, 6.0];
    const expected = [
      0.0,
      0.0,
      (-1.5 * (-1.5 + 3.0)) / 6.0, // -0.375
      0.0,
      (1.0 * (1.0 + 3.0)) / 6.0, // 4.0 / 6.0 = 0.666667
      (2.0 * (2.0 + 3.0)) / 6.0, // 10.0 / 6.0 = 1.666667
      3.0,
      6.0,
    ];

    const inp = ctx.tensor('float32', [1, 8]).setData(new Float32Array(inputs));
    const out = ctx.tensor('float32', [1, 8]);

    ctx.recordShader(shaders.hardswish, [inp, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, expected);
    ctx.destroy();
  });
});
