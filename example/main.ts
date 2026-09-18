import { initDevice, isWebGPUSupported, tensor } from '../src';

const statusEl = document.getElementById('status') as HTMLDivElement;
const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
const logEl = document.getElementById('log') as HTMLPreElement;

function log(msg: string) {
  logEl.textContent += `${msg}\n`;
}

if (!isWebGPUSupported()) {
  statusEl.innerHTML =
    '<span class="tag tag-warn">Unsupported</span> WebGPU is not supported in this browser.';
} else {
  statusEl.innerHTML = '<span class="tag tag-success">Supported</span> WebGPU is available!';
  runBtn.disabled = false;
}

runBtn.addEventListener('click', async () => {
  runBtn.disabled = true;
  logEl.textContent = '';

  try {
    log('1. Initializing WebGPU device...');
    const device = await initDevice();
    log(`   Device acquired!`);
    log(`   - shader-f16 supported: ${device.features.has('shader-f16')}`);
    log(`   - subgroups supported:  ${device.features.has('subgroups')}`);

    log('\n--- Test A: Tensor from TypedArray ---');
    log('2. Creating host input tensor data (Float32Array)...');
    const shapeA = [2, 3] as const;
    const rawDataA = new Float32Array([1.5, -2.0, 3.25, 4.125, -5.5, 6.0]);
    log(`   Shape: [${shapeA.join(', ')}]`);
    log(`   Data:  [${Array.from(rawDataA).join(', ')}]`);

    log('3. Allocating WebGPU Tensor (uploading to VRAM)...');
    const tensorA = tensor('float32', shapeA, rawDataA, device);
    log(`   VRAM buffer allocated (size: ${tensorA.buffer.size} bytes)`);

    log('4. Reading data back from GPU via tensorA.getData()...');
    const arrayBufferA = await tensorA.getData();
    const resultA = new Float32Array(arrayBufferA);
    log(`   Result: [${Array.from(resultA).join(', ')}]`);

    const matchesA = rawDataA.every((val, idx) => Math.abs(val - resultA[idx]) < 1e-6);
    if (matchesA) {
      log('   ✅ Test A PASSED: GPU roundtrip data matches host input exactly!');
    } else {
      log('   ❌ Test A FAILED: GPU readback does not match input!');
    }

    tensorA.destroy();
    log('   Buffer destroyed.');

    log('\n--- Test B: Tensor from existing GPUBuffer (Zero-Copy) ---');
    log('5. Allocating external GPUBuffer directly on device...');
    const shapeB = [4] as const;
    const rawDataB = new Int32Array([10, -20, 30, -40]);
    const byteLengthB = rawDataB.byteLength;

    const externalBuffer = device.createBuffer({
      size: Math.max(16, byteLengthB),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(externalBuffer, 0, rawDataB.buffer, rawDataB.byteOffset, byteLengthB);
    log(`   External GPUBuffer created (size: ${externalBuffer.size} bytes)`);

    log('6. Wrapping external GPUBuffer into Tensor...');
    const tensorB = tensor('int32', shapeB, externalBuffer, device);
    log(
      `   Wrapped zero-copy (tensorB.buffer === externalBuffer: ${tensorB.buffer === externalBuffer})`
    );

    log('7. Reading data back from wrapped GPUBuffer via tensorB.getData()...');
    const arrayBufferB = await tensorB.getData();
    const resultB = new Int32Array(arrayBufferB);
    log(`   Result: [${Array.from(resultB).join(', ')}]`);

    const matchesB = rawDataB.every((val, idx) => val === resultB[idx]);
    if (matchesB) {
      log('   ✅ Test B PASSED: GPUBuffer wrapped and read back correctly!');
    } else {
      log('   ❌ Test B FAILED: Readback does not match input!');
    }

    tensorB.destroy();
    log('   External buffer destroyed via tensorB.destroy().');

    log('\n🎉 All checks completed successfully!');
  } catch (err) {
    log(`\n❌ Error: ${(err as Error).message}`);
  } finally {
    runBtn.disabled = false;
  }
});
