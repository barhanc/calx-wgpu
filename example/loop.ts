/**
 * Per-frame WebGPU loop for the selfie-segmentation demo.
 *
 * Implements a record-once pipeline:
 * 1. Snapshot video frame into persistent video texture (device.queue.copyExternalImageToTexture).
 * 2. Record preprocess shader (samples video texture -> writes input tensor).
 * 3. Record ExecuTorch model program (input tensor -> output mask tensor).
 * 4. Record postprocess shader (blends video texture with mask -> writes result texture).
 * 5. Per-frame submit recorded queue commands and copy result texture to canvas.
 */

import type { Program } from '../src';
import { WgpuExecutionContext } from '../src';
import { postprocess, preprocess } from './shaders';

const size = [256, 256];

function onNextVideoFrame(video: HTMLVideoElement, callback: () => void): void {
  if ('requestVideoFrameCallback' in video) {
    video.requestVideoFrameCallback(() => callback());
  } else {
    requestAnimationFrame(() => callback());
  }
}

/**
 * Starts the per-frame loop for the given camera video.
 *
 * @param video The playing camera video to source frames from.
 * @param canvas The canvas element to render the segmented video output to.
 * @param onMetrics Optional callback invoked after each frame with FPS and latency in ms.
 * @returns A stop function that halts the frame loop and cleans up resources.
 */
export async function startFrameLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  onMetrics?: (fps: number, ms: number) => void
): Promise<() => void> {
  // Acquire WebGPU device
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter?.requestDevice();

  if (device === undefined) {
    throw new Error('no WebGPU device');
  }

  // Configure canvas context
  const canvasCtx = canvas.getContext('webgpu');

  if (canvasCtx === null) {
    throw new Error('failed to get WebGPU context for canvas');
  }

  canvas.width = size[0];
  canvas.height = size[1];

  canvasCtx.configure({
    device,
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    alphaMode: 'premultiplied',
  });

  // Persistent textures — keep bind groups valid across frames
  const vidTexture = device.createTexture({
    size,
    format: 'rgba8unorm',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  });
  const resTexture = device.createTexture({
    size,
    format: 'rgba8unorm',
    usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING,
  });

  // Load model program descriptor and weights
  const [programRes, weightsRes] = await Promise.all([
    fetch('/assets/selfie-segmentation/program.json'),
    fetch('/assets/selfie-segmentation/weights.bin'),
  ]);

  if (!programRes.ok || !weightsRes.ok) {
    throw new Error('failed to fetch selfie segmentation model artifacts');
  }

  const program = (await programRes.json()) as Program;
  const weights = await weightsRes.arrayBuffer();

  // Record the compute pipeline
  const ctx = new WgpuExecutionContext(device);
  const inp = ctx.tensor('float32', [1, 3, ...size]);
  const out = ctx.tensor('float32', [1, 1, ...size]);
  ctx
    .recordShader(preprocess, [vidTexture, inp])
    .recordProgram(program, weights, [inp, out])
    .recordShader(postprocess, [vidTexture, out, resTexture]);

  let running = true;
  let last = performance.now();

  const onFrame = async (): Promise<void> => {
    if (!running) return;

    const start = performance.now();

    // Snapshot video frame into persistent video texture
    device.queue.copyExternalImageToTexture({ source: video }, { texture: vidTexture }, size);

    // Re-encode and submit recorded compute commands
    ctx.submit();

    // Present result texture to canvas
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToTexture(
      { texture: resTexture },
      { texture: canvasCtx.getCurrentTexture() },
      size
    );
    device.queue.submit([encoder.finish()]);

    // Wait for GPU execution to complete and measure pipeline latency
    await device.queue.onSubmittedWorkDone();

    // Track frame rate and report metrics
    const ms = performance.now() - start;
    const now = performance.now();
    const fps = Math.round(1000 / (now - last));

    onMetrics?.(fps, ms);
    last = now;

    onNextVideoFrame(video, onFrame);
  };

  onNextVideoFrame(video, onFrame);

  return () => {
    running = false;
    ctx.destroy();
    vidTexture.destroy();
    resTexture.destroy();
  };
}
