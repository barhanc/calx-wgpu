import { startFrameLoop } from './loop';

const video = document.getElementById('video') as HTMLVideoElement;
const canvas = document.getElementById('view') as HTMLCanvasElement;

const startBtn = document.getElementById('start-btn') as HTMLButtonElement;
const statusDot = document.getElementById('status-dot') as HTMLSpanElement;
const statusText = document.getElementById('status-text') as HTMLSpanElement;

const fps = document.getElementById('fps') as HTMLDivElement;

function setStatus(text: string, state: 'idle' | 'starting' | 'live' | 'error') {
  statusText.textContent = text;
  statusDot.classList.toggle('live', state === 'live');
  statusDot.classList.toggle('error', state === 'error');
  startBtn.disabled = state === 'starting' || state === 'live';
  startBtn.setAttribute('aria-label', state === 'live' ? 'Camera running' : 'Start camera');
}

async function start() {
  setStatus('requesting camera', 'starting');
  try {
    const config = { video: { width: 1280, height: 720 }, audio: false };
    const stream = await navigator.mediaDevices.getUserMedia(config);

    video.srcObject = stream;
    await video.play();

    setStatus('camera live', 'live');
    await startFrameLoop(video, canvas);
  } catch (error) {
    // eslint-disable-next-line no-console
    console.error(error);
    fps.textContent = '';
    setStatus('pipeline error', 'error');
  }
}

startBtn.addEventListener('click', () => void start());
