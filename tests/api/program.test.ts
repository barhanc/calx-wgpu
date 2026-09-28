import { describe, it, expect } from 'vitest';
import { WgpuExecutionContext, type Program } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

// Program: out = (a + 1.5 * b) @ c
//
// Chain:
//   1. aten.add.Tensor: temp = a + 1.5 * b   (values 0,1,2 → 3)
//   2. aten.mm.default:  out = temp @ c      (values 3,4 → 5)
//
// Memory planning (all tensors [2,2] = 16 bytes):
//   - `a` is live in kernel 1 only, `out` is live in kernel 2 only
//     → they share pool 0 (non-overlapping lifetimes)
//   - `b` is live in kernel 1 only → pool 1
//   - `temp` is live in both kernels → pool 2
//   - `c` is live in kernel 2 only → pool 3
//
// Values:
//   [0] a     input        [2,2]  pool 0
//   [1] b     input        [2,2]  pool 1
//   [2] alpha scalar       1.5
//   [3] temp  intermediate [2,2]  pool 2
//   [4] c     input        [2,2]  pool 3
//   [5] out   output       [2,2]  pool 0 (reuses a)
const prog: Program = {
  version: '1',
  chain: [
    { name: 'aten.add.Tensor', args: [0, 1, 2, 3] },
    { name: 'aten.mm.default', args: [3, 4, 5] },
  ],
  values: [
    { type: 'tensor', shape: [2, 2], dtype: 'float32', memObjId: 0 },
    { type: 'tensor', shape: [2, 2], dtype: 'float32', memObjId: 1 },
    { type: 'scalar', value: 1.5 },
    { type: 'tensor', shape: [2, 2], dtype: 'float32', memObjId: 2 },
    { type: 'tensor', shape: [2, 2], dtype: 'float32', memObjId: 3 },
    { type: 'tensor', shape: [2, 2], dtype: 'float32', memObjId: 0 },
  ],
  inputIds: [0, 1, 4],
  outputIds: [5],
  memoryPlan: {
    pools: [
      { id: 0, size: 16 },
      { id: 1, size: 16 },
      { id: 2, size: 16 },
      { id: 3, size: 16 },
    ],
  },
};

const weights = new ArrayBuffer(0);

describe('WgpuExecutionContext.recordProgram()', () => {
  it('runs a 2-kernel program with buffer reuse', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const a = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const b = ctx.tensor('float32', [2, 2]).setData(new Float32Array([10, 20, 30, 40]));
    const c = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const out = ctx.tensor('float32', [2, 2]);

    ctx.recordProgram(prog, weights, [a, b, c, out]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(out, [112, 160, 240, 352]);
    ctx.destroy();
  });

  it('rejects wrong arg count', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const a = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const b = ctx.tensor('float32', [2, 2]).setData(new Float32Array([10, 20, 30, 40]));
    const c = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));

    expect(() => ctx.recordProgram(prog, weights, [a, b, c])) // prettier-ignore
      .toThrow(/expected 4 args/);
    ctx.destroy();
  });

  it('rejects non-Tensor output', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const a = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const b = ctx.tensor('float32', [2, 2]).setData(new Float32Array([10, 20, 30, 40]));
    const c = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));

    expect(() => ctx.recordProgram(prog, weights, [a, b, c, 42])) // prettier-ignore
      .toThrow(/output 0 must be a Tensor/);
    ctx.destroy();
  });

  it('rejects unknown operator', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const badProgram: Program = {
      ...prog,
      chain: [{ name: 'aten.nonexistent.default', args: [0, 1, 2, 3] }],
    };

    const a = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const b = ctx.tensor('float32', [2, 2]).setData(new Float32Array([10, 20, 30, 40]));
    const c = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const out = ctx.tensor('float32', [2, 2]);

    expect(() => ctx.recordProgram(badProgram, weights, [a, b, c, out])) // prettier-ignore
      .toThrow(/unknown operator/);
    ctx.destroy();
  });

  it('rejects missing memory pool', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const badProgram: Program = {
      ...prog,
      values: [
        { type: 'tensor', shape: [2, 2], dtype: 'float32', memObjId: 99 },
        ...prog.values.slice(1),
      ],
    };

    const a = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const b = ctx.tensor('float32', [2, 2]).setData(new Float32Array([10, 20, 30, 40]));
    const c = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const out = ctx.tensor('float32', [2, 2]);

    expect(() => ctx.recordProgram(badProgram, weights, [a, b, c, out])) // prettier-ignore
      .toThrow(/no memory pool 99/);
    ctx.destroy();
  });
});
