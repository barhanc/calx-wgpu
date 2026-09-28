import { describe, it, expect } from 'vitest';
import { WgpuExecutionContext, Tensor } from '../../src';
import { setupGPU } from '../helpers';

describe('WgpuExecutionContext', () => {
  it('allocates storage buffers', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const buf = ctx.storageBuffer(256);
    expect(buf.size).toBeGreaterThanOrEqual(256);
    expect(buf.usage & GPUBufferUsage.STORAGE).toBeTruthy();
    ctx.destroy();
  });

  it('aligns storage buffer size to 4 bytes', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const buf = ctx.storageBuffer(1);
    expect(buf.size).toBe(4);
    ctx.destroy();
  });

  it('creates tensors with owned buffers', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const t = ctx.tensor('float32', [2, 3]);
    expect(t.shape).toEqual([2, 3]);
    expect(t.byteLength).toBe(24);
    ctx.destroy();
  });

  it('copy() validates COPY_SRC on source', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const srcBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const dst = ctx.tensor('float32', [4]);
    const src = new Tensor('float32', [4], device, srcBuf);
    expect(() => ctx.copy(src, dst)).toThrow(/COPY_SRC/);
    srcBuf.destroy();
    ctx.destroy();
  });

  it('copy() validates COPY_DST on destination', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const src = ctx.tensor('float32', [4]);
    const dstBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const dst = new Tensor('float32', [4], device, dstBuf);
    expect(() => ctx.copy(src, dst)).toThrow(/COPY_DST/);
    dstBuf.destroy();
    ctx.destroy();
  });

  it('copy() validates destination >= source size', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const src = ctx.tensor('float32', [4]);
    const dst = ctx.tensor('float32', [2]);
    expect(() => ctx.copy(src, dst)).toThrow(/< source/);
    ctx.destroy();
  });

  it('copy() + submit() transfers data between buffers', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const src = ctx.tensor('float32', [4]);
    const dst = ctx.tensor('float32', [4]);
    src.setData(new Float32Array([1, 2, 3, 4]));

    ctx.copy(src, dst);
    ctx.submit();
    await ctx.sync();

    const data = new Float32Array(await dst.getData());
    expect(Array.from(data)).toEqual([1, 2, 3, 4]);
    ctx.destroy();
  });

  it('destroy() prevents further use', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    ctx.destroy();
    expect(() => ctx.storageBuffer(16)).toThrow(/destroyed/);
  });

  it('destroy() is idempotent', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    ctx.destroy();
    expect(() => ctx.destroy()).not.toThrow();
  });

  it('submit() with no commands is a no-op', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    expect(() => ctx.submit()).not.toThrow();
    ctx.destroy();
  });

  it('sync() resolves after submit()', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const src = ctx.tensor('float32', [4]).setData(new Float32Array([1, 2, 3, 4]));
    const dst = ctx.tensor('float32', [4]);
    ctx.copy(src, dst);
    ctx.submit();
    await expect(ctx.sync()).resolves.toBeUndefined();
    ctx.destroy();
  });

  it('sync() with no prior submit() resolves', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    await expect(ctx.sync()).resolves.toBeUndefined();
    ctx.destroy();
  });

  it('sync() on destroyed context throws', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    ctx.destroy();
    await expect(ctx.sync()).rejects.toThrow(/destroyed/);
  });
});
