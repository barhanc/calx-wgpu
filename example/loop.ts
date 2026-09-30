/**
 * Per-frame WebGPU loop for the selfie-segmentation demo.
 *
 * Owns the device and a per-video-frame callback. Every frame is imported as a
 * GPUExternalTexture — the source the model-input shader will sample in a
 * later step.
 */

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
 * @returns A stop function that halts the frame loop.
 */
export async function startFrameLoop(video: HTMLVideoElement): Promise<() => void> {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter?.requestDevice();
  if (device === undefined) {
    throw new Error('no WebGPU device');
  }

  let running = true;
  const onFrame = (): void => {
    if (!running) return;

    // External textures are frame-scoped and must be re-imported every frame.
    device.importExternalTexture({ source: video });

    onNextVideoFrame(video, onFrame);
  };

  onNextVideoFrame(video, onFrame);
  return () => (running = false);
}
