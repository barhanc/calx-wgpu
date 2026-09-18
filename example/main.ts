import { initDevice, isWebGPUSupported, kernels, tensor, WgpuExecutionContext } from '../src';

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
    const tensorA = tensor('float32', shapeA, device, rawDataA);
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
    const tensorB = tensor('int32', shapeB, device, externalBuffer);
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

    log('\n--- Test C: Context Execution of aten.add.Tensor ---');
    log('8. Building WgpuExecutionContext with input tensors and alpha constant...');
    // x shape: [2, 3], y shape: [1, 3] -> broadcast output: [2, 3]
    const xData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const yData = new Float32Array([10.0, 20.0, 30.0]);
    const tensorX = tensor('float32', [2, 3], device, xData);
    const tensorY = tensor('float32', [1, 3], device, yData);
    const tensorOut = tensor('float32', [2, 3], device);

    const ctx = new WgpuExecutionContext(device);
    ctx.setTensor(0, tensorX);
    ctx.setTensor(1, tensorY);
    ctx.setScalar(2, 2.0); // alpha = 2.0
    ctx.setTensor(3, tensorOut);

    // Build and record dispatch into context
    kernels.add.attachTo(ctx, { in1: 0, in2: 1, alpha: 2, out: 3 });
    log('   Context recorded 1 compute dispatch.');

    log('9. Executing context dispatches on WebGPU...');
    const t0 = performance.now();
    ctx.execute();
    await device.queue.onSubmittedWorkDone();
    const durationMs = performance.now() - t0;
    log(`   Execution completed in ${durationMs.toFixed(3)}ms (GPU queue wall-time)`);
    log(`   Out shape: [${tensorOut.shape.join(', ')}]`);

    log('10. Reading back compute shader output...');
    const outBytes = await tensorOut.getData();
    const outFloats = new Float32Array(outBytes);
    log(`   Result: [${Array.from(outFloats).join(', ')}]`);

    // Expected:
    // [1 + 20, 2 + 40, 3 + 60, 4 + 20, 5 + 40, 6 + 60] = [21, 42, 63, 24, 45, 66]
    const expected = [21.0, 42.0, 63.0, 24.0, 45.0, 66.0];
    const matchesC = expected.every((val, idx) => Math.abs(val - outFloats[idx]) < 1e-4);
    if (matchesC) {
      log('   ✅ Test C PASSED: WGSL compute shader add with broadcasting verified!');
    } else {
      log(
        `   ❌ Test C FAILED: Expected [${expected.join(', ')}] but got [${Array.from(outFloats).join(', ')}]`
      );
    }

    ctx.destroy();
    tensorX.destroy();
    tensorY.destroy();
    tensorOut.destroy();

    log('\n🎉 All checks completed successfully!');
  } catch (err) {
    log(`\n❌ Error: ${(err as Error).message}`);
  } finally {
    runBtn.disabled = false;
  }
});
