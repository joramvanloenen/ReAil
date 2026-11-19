import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  makeSphereFaces,
  makeHeadFaces,
  initialBlobNodes,
  makeBlobFaces,
  makeParticleFaces
} from './shape-sources.js';

const $ = (id) => document.getElementById(id);
const sceneCanvas = $('scene-canvas');
const strokeCanvas = $('stroke-canvas');
const viewport = $('scene-viewport');
const output = $('ai-canvas');
const outputCtx = output.getContext('2d', { alpha: false });
const gestureCtx = strokeCanvas.getContext('2d');

// The AI image is also a GPU texture. The UVs are projected from the exact
// camera pose used for the source frame and then remain attached to each mesh.
const feedbackTexture = new THREE.CanvasTexture(output);
feedbackTexture.colorSpace = THREE.SRGBColorSpace;
feedbackTexture.wrapS = THREE.ClampToEdgeWrapping;
feedbackTexture.wrapT = THREE.ClampToEdgeWrapping;
feedbackTexture.minFilter = THREE.LinearFilter;
feedbackTexture.magFilter = THREE.LinearFilter;
feedbackTexture.generateMipmaps = false;
const feedbackUniforms = {
  aiFeedbackTexture: {value: feedbackTexture},
  aiFeedbackStrength: {value: 0}
};
let feedbackReady = false;
let pendingProjection = null;
const remoteCapture = document.createElement('canvas');
remoteCapture.width = 384; remoteCapture.height = 384;
const remoteCaptureCtx = remoteCapture.getContext('2d');
const probeCanvas = document.createElement('canvas');
probeCanvas.width = 32; probeCanvas.height = 32;
const probeCtx = probeCanvas.getContext('2d', {willReadFrequently:true});
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f0f0f);
scene.fog = new THREE.FogExp2(0x0f0f0f, 0.034);
const renderer = new THREE.WebGLRenderer({ canvas: sceneCanvas, antialias: true, powerPreference: 'high-performance', alpha: false });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.36;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 70);
camera.position.set(0, 1.2, 5.65);
const controls = new OrbitControls(camera, sceneCanvas);
controls.target.set(0, 0.02, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 3.1; controls.maxDistance = 11;
controls.enabled = false;
scene.add(new THREE.HemisphereLight(0xd8d8d8, 0x242424, 2.0));
const key = new THREE.DirectionalLight(0xffffff, 3.0);
key.position.set(-3.5, 6.5, 4.7);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
key.shadow.camera.left = -5; key.shadow.camera.right = 5;
key.shadow.camera.top = 5; key.shadow.camera.bottom = -5;
key.shadow.bias = -0.0005;
scene.add(key);
const rim = new THREE.DirectionalLight(0xb8b8b8, 2.2);
rim.position.set(3, 2.8, -4);
scene.add(rim);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(4.1, 96),
  new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.9, metalness: 0 })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -1.62;
floor.receiveShadow = true;
scene.add(floor);
const ring = new THREE.Mesh(
  new THREE.RingGeometry(3.7, 3.715, 120),
  new THREE.MeshBasicMaterial({ color: 0x555555, side: THREE.DoubleSide, transparent: true, opacity: 0.18 })
);
ring.rotation.x = -Math.PI / 2;
ring.position.y = -1.617;
scene.add(ring);

const root = new THREE.Group();
scene.add(root);
const blobHandleGroup = new THREE.Group();
root.add(blobHandleGroup);
const blobHandleGeometry = new THREE.SphereGeometry(0.086, 14, 10);
const blobHandleMaterial = new THREE.MeshBasicMaterial({
  color: 0xbcbcbc, depthTest: false, transparent: true, opacity: 0.82
});
const blobHandleSelectedMaterial = new THREE.MeshBasicMaterial({
  color: 0xf9f9f9, depthTest: false, transparent: true, opacity: 1
});
const materials = [
  new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.33, metalness: 0.01, vertexColors: true, side: THREE.DoubleSide }),
  new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.85, vertexColors: true, side: THREE.DoubleSide })
];

// Modify only the sphere/fragment materials. Lighting and vertex colors stay
// intact; generated AI RGB is mixed into the final shaded output. Custom
// vertex attributes keep per-fragment projected UVs and front-facing masks.
for (const mat of materials) {
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.aiFeedbackTexture = feedbackUniforms.aiFeedbackTexture;
    shader.uniforms.aiFeedbackStrength = feedbackUniforms.aiFeedbackStrength;
    shader.vertexShader = `
      attribute vec2 aiUv;
      attribute float aiVisibility;
      varying vec2 vAiUv;
      varying float vAiVisibility;
    ` + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace(
      '#include <begin_vertex>',
      '#include <begin_vertex>\n vAiUv = aiUv; vAiVisibility = aiVisibility;'
    );

    shader.fragmentShader = `
      uniform sampler2D aiFeedbackTexture;
      uniform float aiFeedbackStrength;
      varying vec2 vAiUv;
      varying float vAiVisibility;
    ` + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <opaque_fragment>',
      `
      if (aiFeedbackStrength > 0.001 && vAiVisibility > 0.001 &&
          vAiUv.x >= 0.0 && vAiUv.x <= 1.0 &&
          vAiUv.y >= 0.0 && vAiUv.y <= 1.0) {
        vec3 aiColor = texture2D(aiFeedbackTexture, vAiUv).rgb;
        outgoingLight = mix(outgoingLight, aiColor,
          clamp(aiFeedbackStrength * vAiVisibility, 0.0, 1.0));
      }
      #include <opaque_fragment>
      `
    );
  };
  mat.customProgramCacheKey = () => 'reail-screen-projection-v1';
}
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let pieces = [];
let cutCount = 0;
let sourceKind = 'sphere';
let blobNodes = initialBlobNodes();
let selectedBlobIndex = 0;
let blobHandleDrag = null;
let lastBlobRebuildAt = 0;
let particles = [];
let particleFaces = null;
let particleEmitAccumulator = 0;
const MAX_PARTICLES = 42;
let lastAutomaticPrompt = $('prompt').value;

const SOURCE_PROMPTS = {
  sphere: 'Photorealistic smooth matte grey sphere, preserve its exact silhouette and camera angle, studio lighting',
  head: 'Photorealistic rough grey clay human head sculpture, prominent nose, eyes, brow, mouth and chin, preserve exact orientation, camera and silhouette',
  blob: 'Photorealistic sculpted neutral grey organic metaball volume with connected lobes, preserve exact silhouette and camera, dramatic studio lighting',
  particles: 'Photorealistic cluster of individual grey floating spherical particles in zero gravity, preserve all sphere positions, size and spacing, black studio background'
};
let rotSpeed = .28;
let rotating = true;
let mode = 'slice';
let maxRate = 10;
let requestAt = 0;
let hasAIFrame = false;
let lastDisplay = null;
let socket = null;
let remotePending = false;
let sequence = 0;
let lastRemoteSent = 0;
let firstInteraction = false;
let lastAIFrame = 0;
let aiCount = 0;
let measuredAiFps = 0;
let prevFrame = performance.now();
let fpsCount = 0;
let sceneFps = 0;
let fpsTick = performance.now();
let lastProbe = null;
let lastProbeSignature = '';
let lastSourceSentAt = 0;
let skippedFrames = 0;

const tuningIds = [
  'diffusion-strength',
  'diffusion-steps',
  'temporal-stability',
  'motion-response',
  'diffusion-seed',
  'lock-seed',
  'inference-resolution',
  'fast-vae',
  'skip-still',
  'feedback-loop',
  'feedback-strength'
];

function updateTuningLabels() {
  const strength = Number($('diffusion-strength').value) / 100;
  const steps = Number($('diffusion-steps').value);
  const stability = Number($('temporal-stability').value);
  const motion = Number($('motion-response').value) / 100;

  $('diffusion-strength-label').textContent = strength.toFixed(2);
  $('diffusion-steps-label').textContent = String(steps);
  $('temporal-stability-label').textContent = Math.round(stability) + '%';
  $('motion-response-label').textContent = motion.toFixed(2) + '×';
}

function enforceValidStrengthForSteps() {
  const steps = Math.max(1, Number($('diffusion-steps').value));
  const minimum = Math.min(1, 1 / steps);
  const slider = $('diffusion-strength');
  if ((Number(slider.value) / 100) < minimum) {
    slider.value = String(Math.ceil(minimum * 100));
  }
  updateTuningLabels();
}

function getRemoteSettings() {
  return {
    strength: Number($('diffusion-strength').value) / 100,
    steps: Number($('diffusion-steps').value),
    temporalStability: Number($('temporal-stability').value) / 100,
    motionResponse: Number($('motion-response').value) / 100,
    resolution: Number($('inference-resolution').value),
    fastVae: $('fast-vae').checked,
    seed: Math.max(0, Math.min(2147483647, Math.floor(Number($('diffusion-seed').value) || 0))),
    lockSeed: $('lock-seed').checked
  };
}

function saveRemoteTuning() {
  try {
    localStorage.setItem('reail-diffusion-settings-sdxl-v1', JSON.stringify({
      ...getRemoteSettings(),
      skipStill: $('skip-still').checked,
      feedbackLoop: $('feedback-loop').checked,
      feedbackStrength: Number($('feedback-strength').value)/100
    }));
  } catch {}
}

function restoreRemoteTuning() {
  try {
    const saved = JSON.parse(localStorage.getItem('reail-diffusion-settings-sdxl-v1') || 'null');
    if (!saved) return;
    if (Number.isFinite(saved.strength)) $('diffusion-strength').value = String(Math.round(saved.strength * 100));
    if (Number.isFinite(saved.steps)) $('diffusion-steps').value = String(Math.round(saved.steps));
    if (Number.isFinite(saved.temporalStability)) $('temporal-stability').value = String(Math.round(saved.temporalStability * 100));
    if (Number.isFinite(saved.motionResponse)) $('motion-response').value = String(Math.round(saved.motionResponse * 100));
    if (Number.isFinite(saved.seed)) $('diffusion-seed').value = String(Math.floor(saved.seed));
    if (typeof saved.lockSeed === 'boolean') $('lock-seed').checked = saved.lockSeed;
    if ([256,384,512].includes(saved.resolution)) $('inference-resolution').value = String(saved.resolution);
    if (typeof saved.fastVae === 'boolean') $('fast-vae').checked = saved.fastVae;
    if (typeof saved.skipStill === 'boolean') $('skip-still').checked = saved.skipStill;
    if (typeof saved.feedbackLoop === 'boolean') $('feedback-loop').checked = saved.feedbackLoop;
    if (Number.isFinite(saved.feedbackStrength)) {
      $('feedback-strength').value = String(Math.round(saved.feedbackStrength*100));
    }
  } catch {}
}

function syncInferenceResolution() {
  const resolution = Number($('inference-resolution').value);
  remoteCapture.width = remoteCapture.height = resolution;
  output.width = output.height = resolution;
  feedbackTexture.needsUpdate = true;
  resetFeedbackProjection();
  updateFeedbackStrength();
  $('resolution-label').textContent = resolution + '²';
  // New resolution must not be compared with the old downsampled source.
  lastProbe = null;
}

function updateRemoteTelemetry(msg) {
  const settings = msg.settings;
  if (!settings) return;

  $('ai-telemetry').hidden = false;
  $('stat-strength').textContent = Number(settings.strength).toFixed(2);
  $('stat-steps').textContent = String(settings.effectiveSteps ?? settings.steps) + '/' + String(settings.steps);
  $('stat-motion').textContent = Number(settings.sceneChange ?? 0).toFixed(2);
  $('stat-gpu').textContent = Number(msg.computeMs || 0).toFixed(0) + 'ms';
  $('stat-server').textContent = Number(msg.serverMs ?? msg.computeMs ?? 0).toFixed(0) + 'ms';
  $('stat-resolution').textContent = String(settings.resolution || '—');
  $('stat-skipped').textContent = String(skippedFrames);
}

function makeGeometry(faces) {
  const pos=[], norms=[], colors=[], counts=[0,0];
  for (let mat=0;mat<2;mat++) {
    for(const f of faces){
      if(f.mat!==mat)continue;
      for (const v of f.v){
        pos.push(v.p.x,v.p.y,v.p.z);
        norms.push(v.n.x,v.n.y,v.n.z);
        colors.push(v.c.r,v.c.g,v.c.b);
      }
      counts[mat] += 3;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  g.setAttribute('normal',new THREE.Float32BufferAttribute(norms,3));
  g.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
  // Every new fragment begins with an invalid projection: no ghosted texture.
  g.setAttribute('aiUv',new THREE.BufferAttribute(
    new Float32Array((pos.length/3)*2).fill(-1),2
  ));
  g.setAttribute('aiVisibility',new THREE.BufferAttribute(
    new Float32Array(pos.length/3),1
  ));
  g.clearGroups();
  g.addGroup(0,counts[0],0);
  g.addGroup(counts[0],counts[1],1);
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}
function snapshotFeedbackProjection() {
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  const viewProjection = new THREE.Matrix4().multiplyMatrices(
    camera.projectionMatrix, camera.matrixWorldInverse
  );
  const cameraPosition = camera.getWorldPosition(new THREE.Vector3());
  const pieceMatrices = new Map();
  for (const piece of pieces) {
    pieceMatrices.set(piece.mesh.uuid, piece.mesh.matrixWorld.clone());
  }
  return {viewProjection, cameraPosition, pieceMatrices};
}

function applyFeedbackProjection(snapshot) {
  if (!snapshot) return;

  const worldPos = new THREE.Vector3();
  const normalMatrix = new THREE.Matrix3();

  for (const piece of pieces) {
    const modelMatrix = snapshot.pieceMatrices.get(piece.mesh.uuid);
    if (!modelMatrix) continue; // Newly cut fragment: wait for next AI result.

    const geometry = piece.mesh.geometry;
    const positions = geometry.getAttribute('position');
    const normals = geometry.getAttribute('normal');
    const uvAttr = geometry.getAttribute('aiUv');
    const maskAttr = geometry.getAttribute('aiVisibility');
    const uv = uvAttr.array;
    const mask = maskAttr.array;

    const clip = new THREE.Matrix4().multiplyMatrices(
      snapshot.viewProjection, modelMatrix
    ).elements;
    const world = modelMatrix.elements;
    normalMatrix.getNormalMatrix(modelMatrix);
    const nm = normalMatrix.elements;
    const cameraPos = snapshot.cameraPosition;

    for (let i=0; i<positions.count; i++) {
      const j=i*3, k=i*2;
      const x=positions.array[j], y=positions.array[j+1], z=positions.array[j+2];

      const w=clip[3]*x+clip[7]*y+clip[11]*z+clip[15];
      if (w<=0.00001) {
        uv[k]=-1; uv[k+1]=-1; mask[i]=0;
        continue;
      }

      const px=(clip[0]*x+clip[4]*y+clip[8]*z+clip[12])/w;
      const py=(clip[1]*x+clip[5]*y+clip[9]*z+clip[13])/w;
      uv[k]=px*0.5+0.5;
      uv[k+1]=py*0.5+0.5;

      // Keep feedback on the source-visible surface, not the far hemisphere.
      worldPos.set(
        world[0]*x+world[4]*y+world[8]*z+world[12],
        world[1]*x+world[5]*y+world[9]*z+world[13],
        world[2]*x+world[6]*y+world[10]*z+world[14]
      );
      const nx=normals.array[j], ny=normals.array[j+1], nz=normals.array[j+2];
      const nnx=nm[0]*nx+nm[3]*ny+nm[6]*nz;
      const nny=nm[1]*nx+nm[4]*ny+nm[7]*nz;
      const nnz=nm[2]*nx+nm[5]*ny+nm[8]*nz;
      const tx=cameraPos.x-worldPos.x;
      const ty=cameraPos.y-worldPos.y;
      const tz=cameraPos.z-worldPos.z;
      const facing=(nnx*tx+nny*ty+nnz*tz) /
        Math.max(0.000001,Math.hypot(nnx,nny,nnz)*Math.hypot(tx,ty,tz));
      mask[i]=THREE.MathUtils.clamp(facing*5,0,1);
    }

    uvAttr.needsUpdate=true;
    maskAttr.needsUpdate=true;
  }
}

function resetFeedbackProjection() {
  feedbackReady = false;
  pendingProjection = null;
  feedbackUniforms.aiFeedbackStrength.value = 0;
  for (const piece of pieces) {
    const mask = piece.mesh.geometry.getAttribute('aiVisibility');
    if (mask) {
      mask.array.fill(0);
      mask.needsUpdate = true;
    }
  }
}

function updateFeedbackStrength() {
  feedbackUniforms.aiFeedbackStrength.value =
    feedbackReady && $('feedback-loop').checked
      ? Number($('feedback-strength').value)/100
      : 0;
  $('feedback-strength-label').textContent =
    Math.round(Number($('feedback-strength').value))+'%';
}

function addPiece(faces, offset, velocity, dynamic) {
  const mesh = new THREE.Mesh(makeGeometry(faces),materials);
  mesh.castShadow=true;
  mesh.receiveShadow=true;
  mesh.position.copy(offset);
  mesh.userData.isSliceable=true;
  root.add(mesh);
  const chunk={faces,mesh,velocity:velocity || new THREE.Vector3(),dynamic:!!dynamic,settled:false};
  pieces.push(chunk);
  return chunk;
}
function clearSourcePieces() {
  for (const piece of pieces) {
    root.remove(piece.mesh);
    piece.mesh.geometry.dispose();
  }
  pieces = [];
  particles = [];
  particleEmitAccumulator = 0;
}

function refreshBlobHandles() {
  blobHandleGroup.clear();
  if (sourceKind !== 'blob') return;
  blobNodes.forEach((node, i) => {
    const handle = new THREE.Mesh(
      blobHandleGeometry,
      i === selectedBlobIndex ? blobHandleSelectedMaterial : blobHandleMaterial
    );
    handle.position.copy(node.position);
    handle.userData.blobIndex = i;
    handle.renderOrder = 10;
    blobHandleGroup.add(handle);
  });
  syncSelectedBlobRadius();
}

function syncSelectedBlobRadius() {
  const node = blobNodes[selectedBlobIndex];
  if (!node) return;
  $('blob-radius').value = String(Math.round(node.radius * 100));
  $('blob-radius-label').textContent = node.radius.toFixed(2);
}

function rebuildBlobMesh() {
  if (sourceKind !== 'blob') return;
  for (const piece of pieces) {
    root.remove(piece.mesh);
    piece.mesh.geometry.dispose();
  }
  pieces = [];
  const faces = makeBlobFaces(blobNodes);
  if (faces.length) {
    addPiece(faces, new THREE.Vector3(), new THREE.Vector3(), false);
  }
  $('mesh-count').textContent = String(blobNodes.length).padStart(2, '0') + ' LOBES';
  pendingProjection = null;
  lastProbe = null;
}

function emitParticle(startup = false) {
  if (particles.length >= MAX_PARTICLES || sourceKind !== 'particles') return;
  if (!particleFaces) particleFaces = makeParticleFaces();
  const radius = 0.080 + Math.random() * 0.080;
  const displacement = new THREE.Vector3(
    (Math.random() - 0.5) * (startup ? 1.30 : 0.28),
    (Math.random() - 0.5) * (startup ? 1.05 : 0.28),
    (Math.random() - 0.5) * (startup ? 0.95 : 0.28)
  );
  const velocity = new THREE.Vector3(
    (Math.random() - 0.5) * 0.25,
    (Math.random() - 0.5) * 0.25,
    (Math.random() - 0.5) * 0.25
  );
  const chunk = addPiece(particleFaces, displacement, velocity, false);
  chunk.mesh.scale.setScalar(radius);
  chunk.mesh.userData.isSliceable = false;
  chunk.radius = radius;
  particles.push(chunk);
  $('mesh-count').textContent = String(particles.length).padStart(2, '0') + ' PARTICLES';
}

function resetCurrentShape() {
  resetFeedbackProjection();
  clearSourcePieces();
  blobHandleGroup.clear();
  cutCount = 0;
  root.rotation.set(0, 0.25, 0);
  firstInteraction = false;
  lastProbe = null;
  particleFaces = null;
  floor.visible = sourceKind !== 'particles';
  ring.visible = sourceKind !== 'particles';

  if (sourceKind === 'sphere') {
    addPiece(makeSphereFaces(), new THREE.Vector3(), new THREE.Vector3(), false);
  } else if (sourceKind === 'head') {
    addPiece(makeHeadFaces(), new THREE.Vector3(), new THREE.Vector3(), false);
  } else if (sourceKind === 'blob') {
    blobNodes = initialBlobNodes();
    selectedBlobIndex = 0;
    rebuildBlobMesh();
    refreshBlobHandles();
  } else if (sourceKind === 'particles') {
    for (let i=0; i<8; i++) emitParticle(true);
  }

  if (sourceKind !== 'blob' && sourceKind !== 'particles') {
    $('mesh-count').textContent = '01 MESH';
  }
  // Do not show the old shape's generated image as if it belonged to the new source.
  outputCtx.fillStyle = '#0b0b0b';
  outputCtx.fillRect(0, 0, output.width, output.height);
  feedbackTexture.needsUpdate = true;
  updateSourceHint();
}

function selectSourceShape(kind) {
  if (!['sphere', 'head', 'blob', 'particles'].includes(kind)) return;
  if ($('prompt').value === lastAutomaticPrompt) {
    $('prompt').value = SOURCE_PROMPTS[kind];
    lastAutomaticPrompt = SOURCE_PROMPTS[kind];
  }

  sourceKind = kind;
  $('source-shape').value = kind;
  $('blob-tools').hidden = kind !== 'blob';
  $('particle-tools').hidden = kind !== 'particles';
  $('tool-edit').hidden = kind !== 'blob';
  $('tool-slice').hidden = kind === 'particles';
  resetCurrentShape();
  setTool(kind === 'blob' ? 'edit' : kind === 'particles' ? 'orbit' : 'slice');
}

function updateSourceHint() {
  const hint = $('scene-hint');
  if (mode === 'orbit') {
    hint.classList.add('hidden');
    return;
  }
  if (mode === 'edit' && sourceKind === 'blob') {
    hint.textContent = 'DRAG WHITE HANDLES TO SCULPT';
  } else {
    hint.textContent = sourceKind === 'head'
      ? 'DRAG ACROSS HEAD TO SLICE'
      : sourceKind === 'blob'
        ? 'DRAG ACROSS BLOB TO SLICE'
        : 'DRAG TO SLICE';
  }
  hint.classList.remove('hidden');
}

function updateParticlePhysics(dt) {
  if (sourceKind !== 'particles') return;
  const rate = Number($('particle-rate').value) / 100;
  const attraction = Number($('particle-attraction').value) / 100;
  const separation = Number($('particle-separation').value) / 100;

  particleEmitAccumulator += dt;
  if (particleEmitAccumulator >= 1 / Math.max(0.1, rate)) {
    particleEmitAccumulator -= 1 / Math.max(0.1, rate);
    emitParticle();
  }

  // Pairwise equal-and-opposite forces: long-range attraction and
  // short-range separation, with no world gravity or floor collisions.
  for (let i=0; i<particles.length; i++) {
    const a=particles[i];
    for (let j=i+1; j<particles.length; j++) {
      const b=particles[j];
      const delta=b.mesh.position.clone().sub(a.mesh.position);
      const distance=Math.max(0.0001,delta.length());
      const dir=delta.divideScalar(distance);
      const minDistance=(a.radius+b.radius)*1.35;
      let force=0;
      if (distance < minDistance) {
        force = -separation * 2.2 * (minDistance - distance) / minDistance;
      } else {
        force = attraction * 0.16 * Math.min(distance - minDistance, 1.4);
      }
      a.velocity.addScaledVector(dir, force * dt);
      b.velocity.addScaledVector(dir, -force * dt);
    }
  }

  for (const p of particles) {
    // Soft confinement to an invisible cloud; no downward acceleration.
    p.velocity.addScaledVector(p.mesh.position, -0.10 * dt);
    p.velocity.multiplyScalar(Math.exp(-0.75 * dt));
    p.velocity.clampLength(0, 0.62);
    p.mesh.position.addScaledVector(p.velocity, dt);

    // Positional correction keeps the spheres from visibly interpenetrating.
    for (const other of particles) {
      if (other === p) continue;
      const difference=p.mesh.position.clone().sub(other.mesh.position);
      const distance=difference.length();
      const clearance=(p.radius+other.radius)*1.02;
      if (distance < clearance && distance > 0.0001) {
        p.mesh.position.addScaledVector(difference, (clearance-distance)/(distance*3));
      }
    }
  }
}

function lerpVertex(a,b,t){
  return {
    p:a.p.clone().lerp(b.p,t),
    n:a.n.clone().lerp(b.n,t).normalize(),
    c:a.c.clone().lerp(b.c,t)
  };
}
function clipPoly(poly, n, d, keepPositive) {
  const result=[];
  const direction=keepPositive?1:-1;
  for(let i=0;i<poly.length;i++){
    const a=poly[i],b=poly[(i+1)%poly.length];
    const da=(n.dot(a.p)-d)*direction,db=(n.dot(b.p)-d)*direction;
    const ina=da>=-0.00001,inb=db>=-0.00001;
    if(ina)result.push(a);
    if(ina!==inb){
      const t=da/(da-db);
      result.push(lerpVertex(a,b,THREE.MathUtils.clamp(t,0,1)));
    }
  }
  return result;
}
function triangulate(poly,mat,out){
  for(let i=1;i<poly.length-1;i++)out.push(face(poly[0],poly[i],poly[i+1],mat));
}
function cutSurfaceColor(p) {
  const micro = 0.01 * Math.sin(p.x * 15 + p.y * 13) * Math.cos(p.z * 17 - p.y * 9);
  const v = THREE.MathUtils.clamp(0.68 + micro, 0, 1);
  return new THREE.Color().setRGB(v, v, v);
}
function splitFaces(faces,n,d) {
  const positive=[],negative=[],edge=[];
  let posCount=0,negCount=0;
  for(const f of faces){
    const values=f.v.map(v=>n.dot(v.p)-d);
    if(values.every(v=>v>=-0.00001)){positive.push(f);posCount++;continue;}
    if(values.every(v=>v<=0.00001)){negative.push(f);negCount++;continue;}
    triangulate(clipPoly(f.v,n,d,true),f.mat,positive);
    triangulate(clipPoly(f.v,n,d,false),f.mat,negative);
    for(let i=0;i<3;i++){
      const j=(i+1)%3;
      if(values[i]*values[j]<-0.0000000001){
        edge.push(f.v[i].p.clone().lerp(f.v[j].p,values[i]/(values[i]-values[j])));
      }
    }
  }
  if(positive.length<15 || negative.length<15 || edge.length<3)return null;
  const unique=[];
  for(const p of edge)if(!unique.some(v=>v.distanceToSquared(p)<0.00002))unique.push(p);
  if(unique.length<3)return null;
  const center = unique.reduce((a,b)=>a.add(b),new THREE.Vector3()).divideScalar(unique.length);
  let u = new THREE.Vector3(0,1,0).cross(n);
  if(u.lengthSq()<.02) u = new THREE.Vector3(1,0,0).cross(n);
  u.normalize();
  const v = new THREE.Vector3().crossVectors(n,u).normalize();
  unique.sort((a,b)=>
    Math.atan2(a.clone().sub(center).dot(v),a.clone().sub(center).dot(u))-
    Math.atan2(b.clone().sub(center).dot(v),b.clone().sub(center).dot(u))
  );
  function addCap(target,normal,flip){
    const c={p:center.clone(),n:normal.clone(),c:cutSurfaceColor(center)};
    for(let i=0;i<unique.length;i++){
      const a=unique[i], b=unique[(i+1)%unique.length];
      const av={p:a.clone(),n:normal.clone(),c:cutSurfaceColor(a)};
      const bv={p:b.clone(),n:normal.clone(),c:cutSurfaceColor(b)};
      target.push(flip?face(c,bv,av,1):face(c,av,bv,1));
    }
  }
  addCap(positive,n.clone().negate(),true);
  addCap(negative,n.clone(),false);
  return [positive,negative];
}
function slice(normalWorld, pointWorld) {
  if(pieces.length>=28) {flashMessage('MAXIMUM FRAGMENTS REACHED');return false;}
  root.updateMatrixWorld(true);
  const worldQuaternion=new THREE.Quaternion();
  root.getWorldQuaternion(worldQuaternion);
  const n=normalWorld.clone().applyQuaternion(worldQuaternion.clone().invert()).normalize();
  const basePoint=root.worldToLocal(pointWorld.clone());
  let sliced=0;
  const before=[...pieces];
  for(const piece of before){
    if(pieces.length>=28)break;
    const d=n.dot(basePoint.clone().sub(piece.mesh.position));
    const result=splitFaces(piece.faces,n,d);
    if(!result)continue;
    const minFaces=Math.min(result[0].length,result[1].length);
    if(minFaces<18)continue;
    root.remove(piece.mesh);
    piece.mesh.geometry.dispose();
    pieces.splice(pieces.indexOf(piece),1);
    const impulse = .072 + Math.min(.1,cutCount*.009);
    const jitter = new THREE.Vector3((Math.random()-.5)*.035,.045,(Math.random()-.5)*.035);
    addPiece(result[0],piece.mesh.position.clone().addScaledVector(n,impulse),
      piece.velocity.clone().addScaledVector(n,.26).add(jitter),true);
    addPiece(result[1],piece.mesh.position.clone().addScaledVector(n,-impulse),
      piece.velocity.clone().addScaledVector(n,-.26).add(jitter),true);
    sliced++;
  }
  if(sliced){
    cutCount++;
    $('mesh-count').textContent=String(pieces.length).padStart(2,'0')+' FRAGMENTS';
    if(!firstInteraction){firstInteraction=true;$('scene-hint').classList.add('hidden');}
  }else flashMessage('DRAW A STROKE THROUGH THE SPHERE');
  return sliced>0;
}
let noticeTimeout=null;
function flashMessage(text){
  const hint=$('scene-hint');
  hint.textContent=text;
  hint.classList.remove('hidden');
  clearTimeout(noticeTimeout);
  noticeTimeout=setTimeout(()=>{hint.classList.add('hidden');hint.innerHTML='<span class="crosshair">✳</span> DRAG ACROSS SPHERE TO SLICE';},1600);
}
function autoSlice(){
  const angle=cutCount*Math.PI*.27;
  const normal=new THREE.Vector3(Math.cos(angle),Math.sin(angle*.6)*.6,Math.sin(angle)).normalize();
  root.updateWorldMatrix(true,false);
  const point=root.localToWorld(new THREE.Vector3(0,(Math.random()-.5)*.25,0));
  slice(normal,point);
}
function updatePhysics(dt){
  for(const piece of pieces){
    if(!piece.dynamic||piece.settled)continue;
    piece.velocity.y-=1.8*dt;
    piece.mesh.position.addScaledVector(piece.velocity,dt);
    piece.velocity.multiplyScalar(Math.pow(.995,dt*60));
    const minY=piece.mesh.geometry.boundingBox.min.y+piece.mesh.position.y;
    const floorY=-1.61;
    if(minY<floorY){
      piece.mesh.position.y+=floorY-minY;
      if(piece.velocity.y<-.17)piece.velocity.y*=-.12;
      else {piece.velocity.y=0;piece.velocity.x*=.65;piece.velocity.z*=.65;}
      if(piece.velocity.lengthSq()<.001)piece.settled=true;
    }
  }
}
function resize(){
  const r=viewport.getBoundingClientRect();
  if(r.width<5||r.height<5)return;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.6));
  renderer.setSize(r.width,r.height,false);
  camera.aspect=r.width/r.height;
  camera.updateProjectionMatrix();
  strokeCanvas.width=Math.round(r.width*(devicePixelRatio||1));
  strokeCanvas.height=Math.round(r.height*(devicePixelRatio||1));
  gestureCtx.setTransform(devicePixelRatio||1,0,0,devicePixelRatio||1,0,0);
}
window.addEventListener('resize',resize);
new ResizeObserver(resize).observe(viewport);
function normalizedScreen(x,y){
  const r=sceneCanvas.getBoundingClientRect();
  pointer.set((x-r.left)/r.width*2-1,-(y-r.top)/r.height*2+1);
  return pointer;
}
function strokeStartEnd(start,end){
  const r=sceneCanvas.getBoundingClientRect();
  gestureCtx.clearRect(0,0,r.width,r.height);
  if(!start||!end)return;
  gestureCtx.strokeStyle='#d0d0d0';
  gestureCtx.lineWidth=2.1;
  gestureCtx.setLineDash([10,6]);
  gestureCtx.beginPath();
  gestureCtx.moveTo(start.x-r.left,start.y-r.top);
  gestureCtx.lineTo(end.x-r.left,end.y-r.top);
  gestureCtx.stroke();
  gestureCtx.setLineDash([]);
  gestureCtx.fillStyle='#e5e5e5';
  for(const p of [start,end]){
    gestureCtx.beginPath();gestureCtx.arc(p.x-r.left,p.y-r.top,3,0,Math.PI*2);gestureCtx.fill();
  }
}
let dragStart=null;
sceneCanvas.addEventListener('pointerdown',e=>{
  if(mode!=='slice'||e.button!==0)return;
  e.preventDefault();
  sceneCanvas.setPointerCapture(e.pointerId);
  dragStart={x:e.clientX,y:e.clientY};
});
sceneCanvas.addEventListener('pointermove',e=>{
  if(!dragStart)return;
  strokeStartEnd(dragStart,{x:e.clientX,y:e.clientY});
});
function finishStroke(e){
  if(!dragStart)return;
  const start=dragStart;dragStart=null;
  strokeStartEnd(null,null);
  const dx=e.clientX-start.x,dy=e.clientY-start.y;
  if(Math.hypot(dx,dy)<25)return;
  const right=new THREE.Vector3(1,0,0).applyQuaternion(camera.quaternion);
  const up=new THREE.Vector3(0,1,0).applyQuaternion(camera.quaternion);
  const normal=right.multiplyScalar(-dy).addScaledVector(up,dx).normalize();
  normalizedScreen((e.clientX+start.x)*.5,(e.clientY+start.y)*.5);
  raycaster.setFromCamera(pointer,camera);
  const hits=raycaster.intersectObjects(pieces.map(p=>p.mesh),false);
  root.updateWorldMatrix(true,false);
  const through=hits[0]?.point.clone() || root.localToWorld(new THREE.Vector3());
  slice(normal,through);
}
sceneCanvas.addEventListener('pointerup',finishStroke);
sceneCanvas.addEventListener('pointercancel',()=>{dragStart=null;strokeStartEnd(null,null);});

function setTool(which){
  mode=which;
  controls.enabled=which==='orbit';
  sceneCanvas.style.cursor=which==='orbit'?'grab':'crosshair';
  for(const t of ['slice','orbit']){
    $('tool-'+t).classList.toggle('active',t===which);
    $('tool-'+t).setAttribute('aria-pressed',String(t===which));
  }
  $('scene-hint').classList.toggle('hidden',which==='orbit');
  if(which==='slice')$('scene-hint').innerHTML='<span class="crosshair">✳</span> DRAG ACROSS SPHERE TO SLICE';
}
$('tool-slice').addEventListener('click',()=>setTool('slice'));
$('tool-orbit').addEventListener('click',()=>setTool('orbit'));
$('reset').addEventListener('click',resetSphere);
$('rotation').addEventListener('input',e=>{
  rotSpeed=Number(e.target.value)/100;
  $('rotation-label').textContent=rotSpeed.toFixed(2);
});
$('auto-rotate').addEventListener('change',e=>rotating=e.target.checked);
$('rate').addEventListener('input',e=>{
  maxRate=Number(e.target.value);
  $('rate-label').textContent=maxRate+' FPS';
});

for (const id of tuningIds) {
  $(id).addEventListener('input', () => {
    if (id === 'diffusion-steps') enforceValidStrengthForSteps();
    else updateTuningLabels();
    if (id === 'inference-resolution') syncInferenceResolution();
    if (id === 'feedback-loop' || id === 'feedback-strength') updateFeedbackStrength();
    saveRemoteTuning();
  });
  $(id).addEventListener('change', saveRemoteTuning);
}

$('randomize-seed').addEventListener('click', () => {
  const values = new Uint32Array(1);
  crypto.getRandomValues(values);
  $('diffusion-seed').value = String(values[0] % 2147483647);
  saveRemoteTuning();
});

$('reset-ai-memory').addEventListener('click', () => {
  if (socket?.readyState === WebSocket.OPEN) {
    lastProbe = null;
    resetFeedbackProjection();
    socket.send(JSON.stringify({type:'reset'}));
    setBadge('RESETTING','pending');
  } else {
    flashMessage('CONNECT DIFFUSION FIRST');
  }
});

function showOverlay(title,subtitle,error=false){
  $('ai-overlay').hidden=false;
  $('ai-overlay').classList.toggle('error',error);
  $('overlay-title').textContent=title;
  $('overlay-sub').textContent=subtitle;
}
function hideOverlay(){$('ai-overlay').hidden=true;}
function setBadge(label,state=''){
  $('ai-badge').textContent=label;
  $('ai-badge').className='status '+state;
}
function drawOutput(source){
  if(!source)return;
  outputCtx.imageSmoothingEnabled=true;
  outputCtx.clearRect(0,0,output.width,output.height);
  outputCtx.drawImage(source,0,0,output.width,output.height);
  feedbackTexture.needsUpdate = true;
  lastDisplay=source;
}
function noteAIFrame(ms) {
  hasAIFrame=true;
  hideOverlay();
  setBadge('ACTIVE','active');
  aiCount++;
  const now=performance.now();
  if(lastAIFrame){
    const diff=now-lastAIFrame;
    measuredAiFps=measuredAiFps ? measuredAiFps*.78+(1000/diff)*.22 : 1000/diff;
  }
  lastAIFrame=now;
  $('ai-fps').textContent=(measuredAiFps||0).toFixed(1)+' FPS';
  $('ai-latency').textContent=Math.round(ms)+' MS';
}
function stopSocket(){
  if(socket){
    socket.onclose=null;
    socket.close();
    socket=null;
  }
  remotePending=false;
}

function connectRemote(){
  stopSocket();
  resetFeedbackProjection();
  lastProbe=null;
  lastProbeSignature='';
  skippedFrames=0;
  hasAIFrame=false;
  setBadge('CONNECTING','pending');

  let address=$('socket-url').value.trim();
  if(!address) address='ws://127.0.0.1:8765/ws';
  if(/^ws:\/\/127\.0\.0\.1:8765\/?$/.test(address)) address='ws://127.0.0.1:8765/ws';
  $('socket-url').value=address;

  showOverlay('CONNECTING',address);

  try{
    socket=new WebSocket(address);

    const connectTimeout=setTimeout(()=>{
      if(socket && socket.readyState===WebSocket.CONNECTING){
        try{socket.close();}catch{}
        setBadge('OFFLINE','error');
        showOverlay('CONNECTION FAILED','No WebSocket response from '+address,true);
      }
    },5000);

    socket.onopen=()=>{
      clearTimeout(connectTimeout);
      setBadge('CONNECTED','connected');
      $('ai-telemetry').hidden=false;
      $('overlay-title').textContent='WAITING FOR FRAME';
      $('overlay-sub').textContent='';
    };

    socket.onerror=()=>{
      clearTimeout(connectTimeout);
      setBadge('OFFLINE','error');
      showOverlay('CONNECTION FAILED',address,true);
      remotePending=false;
    };

    socket.onclose=()=>{
      clearTimeout(connectTimeout);
      setBadge('OFFLINE','error');
      pendingProjection=null;
      remotePending=false;
    };

    socket.onmessage=(event)=>{
      let msg;
      try{msg=JSON.parse(event.data);}catch{return;}
      if(msg.type!=='frame') remotePending=false;

      if(msg.type==='error'){
        pendingProjection=null;
        setBadge('ERROR','error');
        showOverlay('DIFFUSION ERROR',msg.message||'Unknown server error',true);
        return;
      }

      if(msg.type==='reset'){
        setBadge('CONNECTED','connected');
        $('stat-motion').textContent='—';
        return;
      }

      if(msg.type!=='frame'||!msg.image){
        pendingProjection=null;
        remotePending=false;
        return;
      }

      // Ignore an in-flight response from the resolution that was selected
      // before the user moved the AI resolution control.
      if (msg.settings?.resolution &&
          msg.settings.resolution !== Number($('inference-resolution').value)) {
        pendingProjection=null;
        remotePending=false;
        return;
      }

      updateRemoteTelemetry(msg);
      const sentAt = lastRemoteSent;
      const projection = pendingProjection?.frameId === msg.frameId
        ? pendingProjection.snapshot : null;
      const connection = socket;
      const picture=new Image();
      picture.onload=()=>{
        if (socket!==connection) return;
        // Finish applying the AI result BEFORE allowing the next request:
        // the next source render must already contain this feedback texture.
        drawOutput(picture);
        applyFeedbackProjection(projection);
        feedbackReady=true;
        updateFeedbackStrength();
        noteAIFrame(performance.now()-sentAt);
        pendingProjection=null;
        remotePending=false;
      };
      picture.onerror=()=>{
        pendingProjection=null;
        remotePending=false;
        showOverlay('INVALID IMAGE','Could not decode the AI response.',true);
      };
      picture.src=msg.image;
    };
  }catch(err){
    setBadge('OFFLINE','error');
    showOverlay('CONNECTION FAILED',String(err),true);
  }
}
$('connect').addEventListener('click',connectRemote);
function animate(time) {
  requestAnimationFrame(animate);

  const dt=Math.min(.05,(time-prevFrame)/1000||.016);
  prevFrame=time;

  if(rotating)root.rotation.y+=dt*rotSpeed;
  updatePhysics(dt);
  controls.update();
  renderer.render(scene,camera);

  fpsCount++;
  if(time-fpsTick>=1000){
    sceneFps=Math.round(fpsCount*1000/(time-fpsTick));
    fpsCount=0;
    fpsTick=time;
    $('scene-fps').textContent=sceneFps+' FPS';
  }

  if(socket?.readyState===WebSocket.OPEN && !remotePending && time-requestAt>=1000/maxRate){
    requestAt=time;

    const settings=getRemoteSettings();
    const signature=$('prompt').value + JSON.stringify(settings);
    let shouldSend=true;

    // A symmetric grey sphere can render virtually identical frames for
    // long periods. Avoid re-running SDXL on those frames.
    if($('skip-still').checked){
      probeCtx.drawImage(sceneCanvas,0,0,32,32);
      const sample=probeCtx.getImageData(0,0,32,32).data;

      if(lastProbe && signature===lastProbeSignature && time-lastSourceSentAt<2500){
        let difference=0;
        for(let i=0;i<sample.length;i+=4){
          difference+=Math.abs(sample[i]-lastProbe[i])
            +Math.abs(sample[i+1]-lastProbe[i+1])
            +Math.abs(sample[i+2]-lastProbe[i+2]);
        }
        shouldSend=(difference/(32*32*3))>=1.0;
      }

      if(shouldSend) lastProbe=new Uint8ClampedArray(sample);
    }

    if(!shouldSend){
      skippedFrames++;
      $('stat-skipped').textContent=String(skippedFrames);
    }else{
      lastProbeSignature=signature;
      lastSourceSentAt=time;
      remotePending=true;
      lastRemoteSent=performance.now();

      try{
        const frameId=++sequence;
        const snapshot=snapshotFeedbackProjection();
        remoteCaptureCtx.drawImage(sceneCanvas,0,0,remoteCapture.width,remoteCapture.height);
        pendingProjection={frameId,snapshot};
        socket.send(JSON.stringify({
          type:'frame',
          image:remoteCapture.toDataURL('image/jpeg',.80),
          prompt:$('prompt').value,
          frameId,
          settings
        }));
      }catch(err){
        pendingProjection=null;
        remotePending=false;
        console.error('ReAil frame send failed:',err);
      }
    }
  }
}

restoreRemoteTuning();
enforceValidStrengthForSteps();
syncInferenceResolution();
updateFeedbackStrength();
resetSphere();
resize();
setBadge('OFFLINE');
setTimeout(connectRemote,250);
requestAnimationFrame(animate);