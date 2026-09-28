import type { WgpuExecutionContext } from '../../src';

export type BenchResult = {
  wgpuMs: number;
  iterations: number;
};

export type SpeedReport = {
  wgpuMs: number;
  gpuMs: number;
  cpuMs: number;
};

/**
 * Benchmarks a recorded dispatch by submitting it `iterations` times and
 * measuring wall-clock GPU time.
 *
 * @param ctx The context with recorded commands.
 * @param iterations Number of submit() calls (default: 10).
 * @returns Average WGPU execution time in milliseconds.
 */
export async function benchmark(ctx: WgpuExecutionContext, iterations = 10): Promise<BenchResult> {
  // Warm-up
  ctx.submit();
  await ctx.sync();

  // Measure
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    ctx.submit();
  }
  await ctx.sync();
  const elapsed = performance.now() - start;

  return { wgpuMs: elapsed / iterations, iterations };
}

/**
 * Reports speed comparison: `pytorch / webgpu` (higher = webgpu faster).
 *
 * @param label Test case label.
 * @param timings Timing values in milliseconds.
 */
export function reportSpeed(label: string, timings: SpeedReport): void {
  const { wgpuMs, gpuMs, cpuMs } = timings;
  const vsGpu = gpuMs / wgpuMs;
  const vsCpu = cpuMs / wgpuMs;
  // eslint-disable-next-line no-console
  console.log(
    `  ${label}: ${vsGpu.toFixed(2)}x vs gpu (${wgpuMs.toFixed(3)}ms wgpu / ${gpuMs.toFixed(3)}ms gpu), ` +
      `${vsCpu.toFixed(2)}x vs cpu (${wgpuMs.toFixed(3)}ms wgpu / ${cpuMs.toFixed(3)}ms cpu)`
  );
}
