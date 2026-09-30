/**
 * Per-frame WebGPU loop for the selfie-segmentation demo.
 *
 * Implements a record-once pipeline:
 * 1. Snapshot centered square from video into persistent video texture.
 * 2. Record preprocess shader (samples video texture -> writes input tensor).
 * 3. Record ExecuTorch model program (input tensor -> output mask tensor).
 * 4. Record postprocess shader (blends video texture with mask -> writes result texture).
 * 5. Per-frame copy result texture to canvas via copyTextureToTexture.
 */

import type { Program } from '../../src';
import { WgpuExecutionContext } from '../../src';
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
 * @returns A stop function that halts the frame loop and cleans up resources.
 */
export async function startFrameLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement
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

  // Use rgba8unorm so the canvas texture supports COPY_DST (bgra8unorm does not on some backends)
  canvasCtx.configure({
    device,
    format: 'rgba8unorm',
    alphaMode: 'premultiplied',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
  });

  // Persistent textures — keep bind groups valid across frames
  const vidUsage =
    GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
  const resUsage = GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING;

  const vidTexture = device.createTexture({ size, format: 'rgba8unorm', usage: vidUsage });
  const resTexture = device.createTexture({ size, format: 'rgba8unorm', usage: resUsage });

  // OffscreenCanvas used to crop the centred square from the raw video and scale it to
  // model input size. copyExternalImageToTexture is pixel-for-pixel (no scaling), so we
  // use drawImage to perform the crop + downscale before uploading to the GPU texture.
  const cropCanvas = new OffscreenCanvas(size[0], size[1]);
  const cropCtx = cropCanvas.getContext('2d')!;

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

  const onFrame = async (): Promise<void> => {
    if (!running) return;

    // Crop and scale the centred square of the video to 256×256 using drawImage,
    // then upload the scaled result to the GPU texture.
    const minDim = Math.min(video.videoWidth, video.videoHeight);
    const sx = Math.floor((video.videoWidth - minDim) / 2);
    const sy = Math.floor((video.videoHeight - minDim) / 2);

    cropCtx.drawImage(video, sx, sy, minDim, minDim, 0, 0, size[0], size[1]);
    device.queue.copyExternalImageToTexture({ source: cropCanvas }, { texture: vidTexture }, size);

    // Re-encode and submit recorded compute commands
    ctx.submit();

    // Copy result texture to canvas (no intermediate blit shader needed)
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToTexture(
      { texture: resTexture },
      { texture: canvasCtx.getCurrentTexture() },
      size
    );
    device.queue.submit([encoder.finish()]);

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
