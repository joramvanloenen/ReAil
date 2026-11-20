// Dependency-free regression tests for the geometry-aware feedback controller.
// Run: node test-feedback.mjs
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const moduleSource = await readFile(
  new URL('./feedback-controller.js', import.meta.url), 'utf8'
);
// Data URL preserves ESM syntax without requiring a package.json module type.
const {
  ClosedLoopController, FEEDBACK_PRESETS, depthConfidence,
  measureSilhouetteDrift
} = await import(
  'data:text/javascript;base64,' +
  Buffer.from(moduleSource, 'utf8').toString('base64')
);

const N=48;
const depth=new Uint8Array(N*N*4);
const clean=new Uint8ClampedArray(N*N*4);
const spill=new Uint8ClampedArray(N*N*4);
const centerDepth=(250+100/255)/255;

for(let y=0;y<N;y++){
  for(let x=0;x<N;x++){
    const offset=(y*N+x)*4;
    const depthOffset=((N-1-y)*N+x)*4;
    const inside=Math.hypot(x-N/2,y-N/2)<N/3;
    clean[offset+3]=255;
    spill[offset+3]=255;
    if(inside){
      depth[depthOffset]=250;
      depth[depthOffset+1]=100;
      depth[depthOffset+2]=255;
      depth[depthOffset+3]=255;
      for(let c=0;c<3;c++){
        clean[offset+c]=160;
        spill[offset+c]=160;
      }
    } else if(x<Math.floor(N/4)){
      for(let c=0;c<3;c++)spill[offset+c]=160;
    }
  }
}

assert.ok(depthConfidence(depth,N,0.5,0.5,centerDepth*2-1)>0.7,
  'visible front-facing point should preserve texture');
assert.equal(depthConfidence(depth,N,0.5,0.5,centerDepth*2-1+0.03),0,
  'point behind depth buffer should be rejected');
assert.equal(depthConfidence(depth,N,0,0,centerDepth*2-1),0,
  'background should not receive a projected texture');

const cleanMetric=measureSilhouetteDrift(clean,depth,N);
const spillMetric=measureSilhouetteDrift(spill,depth,N);
assert.equal(cleanMetric.drift,0);
assert.ok(spillMetric.drift>cleanMetric.drift+0.05);

const controller=new ClosedLoopController();
controller.observe(clean,depth,N,FEEDBACK_PRESETS.balanced);
const baseline=controller.gain;
for(let i=0;i<5;i++)controller.observe(spill,depth,N,FEEDBACK_PRESETS.balanced);
assert.ok(controller.gain<baseline,'controller must reduce gain when drift grows');
assert.ok(controller.gain>=0.25 && controller.gain<=1.13,'gain stays bounded');
controller.reset();
assert.equal(controller.gain,1);
assert.equal(controller.hasSample,false);

for(const key of ['stable','balanced','mutate']){
  const preset=FEEDBACK_PRESETS[key];
  assert.ok(preset.geometryLock>=0 && preset.geometryLock<=100);
}
console.log('ReAil closed-loop controller regression tests: PASS');
