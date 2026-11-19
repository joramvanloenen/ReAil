import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const sceneCanvas = $('scene-canvas');
const strokeCanvas = $('stroke-canvas');
const viewport = $('scene-viewport');
const output = $('ai-canvas');
const outputCtx = output.getContext('2d', { alpha: false });
const gestureCtx = strokeCanvas.getContext('2d');
const capture = document.createElement('canvas');
capture.width = 224; capture.height = 224;
const captureCtx = capture.getContext('2d', { willReadFrequently: true });
const temp = document.createElement('canvas');
temp.width = 224; temp.height = 224;
const tempCtx = temp.getContext('2d');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b251e);
scene.fog = new THREE.FogExp2(0x1b251e, 0.034);
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
scene.add(new THREE.HemisphereLight(0xdbe6d3, 0x4b3528, 2.15));
const key = new THREE.DirectionalLight(0xffead5, 3.2);
key.position.set(-3.5, 6.5, 4.7);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
key.shadow.camera.left = -5; key.shadow.camera.right = 5;
key.shadow.camera.top = 5; key.shadow.camera.bottom = -5;
key.shadow.bias = -0.0005;
scene.add(key);
const rim = new THREE.DirectionalLight(0xdbeea8, 2.4);
rim.position.set(3, 2.8, -4);
scene.add(rim);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(4.1, 96),
  new THREE.MeshStandardMaterial({ color: 0x313f30, roughness: 0.85, metalness: 0 })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -1.62;
floor.receiveShadow = true;
scene.add(floor);
const ring = new THREE.Mesh(
  new THREE.RingGeometry(3.7, 3.715, 120),
  new THREE.MeshBasicMaterial({ color: 0x7a9167, side: THREE.DoubleSide, transparent: true, opacity: 0.23 })
);
ring.rotation.x = -Math.PI / 2;
ring.position.y = -1.617;
scene.add(ring);

const root = new THREE.Group();
scene.add(root);
const materials = [
  new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.33, metalness: 0.01, vertexColors: true, side: THREE.DoubleSide }),
  new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.85, vertexColors: true, side: THREE.DoubleSide })
];
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let pieces = [];
let cutCount = 0;
let rotSpeed = .28;
let rotating = true;
let mode = 'slice';
let currentEngine = 'onnx';
let outputSize = 96;
let maxRate = 15;
let worker = null;
let workerReady = false;
let workerBusy = false;
let workerVersion = 0;
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
let inferMs = null;
let prevFrame = performance.now();
let fpsCount = 0;
let sceneFps = 0;
let fpsTick = performance.now();
const white = new THREE.Color();

function pointAt(theta, t) {
  const sin = Math.max(0, Math.sin(Math.PI * t));
  const lobe = 1 + .045 * Math.cos(theta * 5 + .32) + .028 * Math.sin(theta * 3 + t * 5);
  const r = .98 * Math.pow(sin, .67) * (.90 + .20 * t) * lobe;
  const topDimple = .24 * Math.exp(-Math.pow((1 - t) / .095, 2));
  const bottomDimple = .11 * Math.exp(-Math.pow(t / .07, 2));
  const y = (2 * t - 1) * 1.08 - topDimple + bottomDimple
    - .035 * Math.cos(theta * 5 + .3) * Math.exp(-Math.pow((1 - t) / .13, 2));
  return new THREE.Vector3(r * Math.cos(theta), y, r * Math.sin(theta));
}
function surfaceVertex(theta, t) {
  const p = pointAt(theta, t);
  const eps = 0.0002;
  const dt = pointAt(theta, Math.min(.9999, t + eps)).sub(pointAt(theta, Math.max(.0001, t - eps)));
  const da = pointAt(theta + eps, t).sub(pointAt(theta - eps, t));
  const n = dt.cross(da).normalize();
  const sun = Math.max(0, Math.cos(theta - 2.4));
  const freckles = Math.sin(theta * 75 + t * 175) * Math.cos(theta * 47 - t * 133);
  const gold = .035 + .21 * Math.pow(Math.abs(t - .75), 3) + .29 * Math.pow(Math.max(0, t - .78), 2);
  const c = new THREE.Color().setRGB(
    Math.min(1, .61 + .20 * sun + gold + freckles * .016),
    Math.min(1, .073 + .07 * sun + gold * 1.1 + freckles * .012),
    Math.min(1, .041 + .025 * sun + gold * .36 + freckles * .007)
  );
  return { p, n, c };
}
function face(a, b, c, mat) { return { v: [a,b,c], mat }; }
function buildAppleFaces() {
  const out = [];
  const around = 88, rows = 55;
  for (let y = 0; y < rows; y++) {
    const t0 = y / rows, t1 = (y + 1) / rows;
    for (let x = 0; x < around; x++) {
      const a0 = (x / around) * Math.PI * 2;
      const a1 = ((x + 1) / around) * Math.PI * 2;
      const a = surfaceVertex(a0,t0), b = surfaceVertex(a1,t0);
      const c = surfaceVertex(a1,t1), d = surfaceVertex(a0,t1);
      if (y !== 0) out.push(face(a,c,b,0));
      if (y !== rows-1) out.push(face(a,d,c,0));
    }
  }
  return out;
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
  g.clearGroups();
  g.addGroup(0,counts[0],0);
  g.addGroup(counts[0],counts[1],1);
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}
function addPiece(faces, offset, velocity, dynamic) {
  const mesh = new THREE.Mesh(makeGeometry(faces),materials);
  mesh.castShadow=true;
  mesh.receiveShadow=true;
  mesh.position.copy(offset);
  mesh.userData.isFruit=true;
  root.add(mesh);
  const chunk={faces,mesh,velocity:velocity || new THREE.Vector3(),dynamic:!!dynamic,settled:false};
  pieces.push(chunk);
  return chunk;
}
let stem = null;
function addStem() {
  stem = new THREE.Group();
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0,.82,0),
    new THREE.Vector3(.016,.99,-.009),
    new THREE.Vector3(.072,1.16,-.035),
    new THREE.Vector3(.11,1.28,-.046)
  ]);
  const twig = new THREE.Mesh(new THREE.TubeGeometry(curve,12,.043,8,false),
    new THREE.MeshStandardMaterial({color:0x634727,roughness:1}));
  twig.castShadow=true;
  stem.add(twig);
  const leafShape = new THREE.Shape();
  leafShape.moveTo(0,0);
  leafShape.bezierCurveTo(.3,.13,.53,.37,.69,.62);
  leafShape.bezierCurveTo(.2,.66,-.04,.31,0,0);
  const leaf = new THREE.Mesh(new THREE.ShapeGeometry(leafShape,20),
    new THREE.MeshStandardMaterial({color:0x56873b,side:THREE.DoubleSide,roughness:.9}));
  leaf.position.set(.08,1.13,-.05);
  leaf.rotation.x = -.52; leaf.rotation.y = -.4; leaf.rotation.z = -.15;
  leaf.castShadow=true;
  stem.add(leaf);
  root.add(stem);
}
function resetApple() {
  for(const p of pieces) {root.remove(p.mesh);p.mesh.geometry.dispose();}
  pieces = []; cutCount = 0;
  if(stem) root.remove(stem);
  addPiece(buildAppleFaces(),new THREE.Vector3(),new THREE.Vector3(),false);
  addStem();
  root.rotation.y = .25;
  $('mesh-count').textContent='01 MESH';
  $('scene-hint').classList.remove('hidden');
  firstInteraction=false;
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
function fleshColor(p) {
  const radial=Math.hypot(p.x,p.z);
  const core = THREE.MathUtils.smoothstep(radial,.085,.32);
  const striation=.025*Math.sin(p.x*19+p.y*33)*Math.sin(p.z*27-p.y*21);
  return new THREE.Color().setRGB(
    .66+.32*core+striation,
    .38+.48*core+striation,
    .21+.58*core+striation
  );
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
    const c={p:center.clone(),n:normal.clone(),c:fleshColor(center)};
    for(let i=0;i<unique.length;i++){
      const a=unique[i], b=unique[(i+1)%unique.length];
      const av={p:a.clone(),n:normal.clone(),c:fleshColor(a)};
      const bv={p:b.clone(),n:normal.clone(),c:fleshColor(b)};
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
    if(stem) {
      const topPiece = pieces.slice().sort((a,b)=>{
        const ac=a.mesh.geometry.boundingBox.max.y+a.mesh.position.y;
        const bc=b.mesh.geometry.boundingBox.max.y+b.mesh.position.y;
        return bc-ac;
      })[0];
      if(topPiece)stem.position.copy(topPiece.mesh.position);
    }
  }else flashMessage('DRAW A STROKE THROUGH THE FRUIT');
  return sliced>0;
}
let noticeTimeout=null;
function flashMessage(text){
  const hint=$('scene-hint');
  hint.textContent=text;
  hint.classList.remove('hidden');
  clearTimeout(noticeTimeout);
  noticeTimeout=setTimeout(()=>{hint.classList.add('hidden');hint.innerHTML='<span class="crosshair">✳</span> DRAG ACROSS APPLE TO SLICE';},1600);
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
  if(stem&&pieces.length>1){
    const topPiece=pieces.reduce((best,p)=>
      !best||p.mesh.geometry.boundingBox.max.y+p.mesh.position.y>best.mesh.geometry.boundingBox.max.y+best.mesh.position.y?p:best,null);
    if(topPiece)stem.position.lerp(topPiece.mesh.position,Math.min(1,dt*8));
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
  gestureCtx.strokeStyle='#d9f887';
  gestureCtx.lineWidth=2.1;
  gestureCtx.setLineDash([10,6]);
  gestureCtx.beginPath();
  gestureCtx.moveTo(start.x-r.left,start.y-r.top);
  gestureCtx.lineTo(end.x-r.left,end.y-r.top);
  gestureCtx.stroke();
  gestureCtx.setLineDash([]);
  gestureCtx.fillStyle='#e5ffa1';
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
  if(which==='slice')$('scene-hint').innerHTML='<span class="crosshair">✳</span> DRAG ACROSS APPLE TO SLICE';
}
$('tool-slice').addEventListener('click',()=>setTool('slice'));
$('tool-orbit').addEventListener('click',()=>setTool('orbit'));
$('slice-once').addEventListener('click',autoSlice);
$('reset').addEventListener('click',resetApple);
$('rotation').addEventListener('input',e=>{rotSpeed=Number(e.target.value)/100;$('rotation-label').textContent=rotSpeed.toFixed(2)+' RAD/S';});
$('auto-rotate').addEventListener('change',e=>rotating=e.target.checked);
$('pixelated').addEventListener('change',e=>output.classList.toggle('smooth',!e.target.checked));
$('resolution').addEventListener('input',e=>{
  const values=[64,96,128,224];
  outputSize=values[Number(e.target.value)];
  $('resolution-label').textContent=outputSize+' × '+outputSize;
  $('output-resolution').textContent=outputSize+' × '+outputSize+' OUTPUT';
  output.width=outputSize;output.height=outputSize;
  if(lastDisplay) drawOutput(lastDisplay);
});
$('rate').addEventListener('input',e=>{maxRate=Number(e.target.value);$('rate-label').textContent=maxRate+' FPS';});
$('show-raw').addEventListener('change',()=>{hasAIFrame=false;});
$('style').addEventListener('change',()=>{if(currentEngine==='onnx')startNeural();});

function showOverlay(title,subtitle,error=false){
  $('ai-overlay').hidden=false;
  $('ai-overlay').classList.toggle('error',error);
  $('overlay-title').textContent=title;
  $('overlay-sub').textContent=subtitle;
}
function hideOverlay(){$('ai-overlay').hidden=true;}
function setBadge(label,state=''){
  $('ai-badge').textContent=label;
  $('ai-badge').className='pill '+state;
}
function drawOutput(source){
  if(!source)return;
  outputCtx.imageSmoothingEnabled=true;
  outputCtx.drawImage(source,0,0,outputSize,outputSize);
  lastDisplay=source;
}
function setModeLabel(label){$('ai-mode-label').textContent=label;}
function noteAIFrame(ms) {
  hasAIFrame=true;
  hideOverlay();
  setBadge('AI ACTIVE');
  inferMs=ms;
  aiCount++;
  const now=performance.now();
  if(lastAIFrame){
    const diff=now-lastAIFrame;
    measuredAiFps=measuredAiFps ? measuredAiFps*.78+(1000/diff)*.22 : 1000/diff;
  }
  lastAIFrame=now;
  $('ai-fps').textContent=(measuredAiFps||0).toFixed(1)+' AI FPS';
  $('ai-latency').textContent='LATENCY '+Math.round(ms)+' MS';
}
function getSourceFrame(){
  captureCtx.drawImage(sceneCanvas,0,0,224,224);
}
function rawPreview(){
  drawOutput(capture);
  if(currentEngine==='preview'||$('show-raw').checked){
    $('ai-fps').textContent=sceneFps+' RENDER FPS';
  }
}
function stopWorker(){
  workerVersion++;
  if(worker){worker.terminate();worker=null;}
  workerReady=false;workerBusy=false;
}
function stopSocket(){
  if(socket){socket.onclose=null;socket.close();socket=null;}
  remotePending=false;
}
function startNeural(){
  stopSocket();stopWorker();
  currentEngine='onnx';
  hasAIFrame=false;
  setBadge('LOADING','pending');
  $('model-corner').textContent='NN_01 · ONNX STYLE';
  setModeLabel('NEURAL / ON-DEVICE');
  $('truth').textContent='Real feed-forward ONNX neural style transfer at 224 × 224. Display resolution does not change inference cost or promise 30 FPS.';
  showOverlay('Loading the neural model','Downloading approximately 6.7 MB. First load can take a moment.');
  const myVersion=workerVersion;
  try {worker=new Worker('./inference-worker.js');}
  catch(err){showOverlay('Worker unavailable',String(err),true);setBadge('AI ERROR','error');return;}
  worker.onmessage=(event)=>{
    if(myVersion!==workerVersion)return;
    const msg=event.data;
    if(msg.type==='ready'){
      workerReady=true;
      setBadge('READY');
      $('overlay-title').textContent='Neural model ready';
      $('overlay-sub').textContent='Generating first interpreted frame...';
    }else if(msg.type==='loading'){
      $('overlay-sub').textContent=msg.message;
    }else if(msg.type==='frame'){
      workerBusy=false;
      const data=new ImageData(new Uint8ClampedArray(msg.pixels),224,224);
      tempCtx.putImageData(data,0,0);
      if(!$('show-raw').checked){
        drawOutput(temp);
        noteAIFrame(performance.now()-msg.sent);
      }
    }else if(msg.type==='error'){
      workerBusy=false;workerReady=false;
      setBadge('AI UNAVAILABLE','error');
      showOverlay('Could not load neural AI',msg.message+' — preview remains available.',true);
      $('truth').textContent='The neural model failed to load. The visible image is now a raw downsampled preview, NOT AI.';
    }
  };
  worker.onerror=(e)=>{
    if(myVersion!==workerVersion)return;
    workerBusy=false;workerReady=false;
    setBadge('AI ERROR','error');
    showOverlay('Neural runtime error',e.message||'Unable to start ONNX Runtime',true);
  };
  worker.postMessage({type:'init',style:$('style').value});
}
function connectRemote(){
  stopWorker();stopSocket();
  currentEngine='remote';hasAIFrame=false;
  setBadge('CONNECTING','pending');
  setModeLabel('DIFFUSION / EXTERNAL GPU');
  $('model-corner').textContent='NN_02 · SD-TURBO';
  $('truth').textContent='Real generative diffusion requires your own running GPU server. Frames are sent from this browser to the server over WebSocket.';
  const address=$('socket-url').value.trim();
  showOverlay('Connecting to diffusion','Requires your own running GPU server.');
  try{
    socket=new WebSocket(address);
    socket.onopen=()=>{setBadge('CONNECTED');$('overlay-title').textContent='Waiting for diffusion frame';$('overlay-sub').textContent='Sending image-to-image requests...';};
    socket.onerror=()=>{setBadge('REMOTE ERROR','error');showOverlay('WebSocket connection failed','Start server.py on your GPU machine; use secure WSS for a public HTTPS site.',true);remotePending=false;};
    socket.onclose=()=>{setBadge('OFFLINE','error');remotePending=false;};
    socket.onmessage=async(event)=>{
      let msg;
      try{msg=JSON.parse(event.data);}catch{return;}
      remotePending=false;
      if(msg.type==='error'){setBadge('INFERENCE ERROR','error');showOverlay('Diffusion server error',msg.message||'Unknown inference failure',true);return;}
      if(msg.type!=='frame'||!msg.image)return;
      const picture=new Image();
      picture.onload=()=>{
        if(currentEngine!=='remote')return;
        drawOutput(picture);
        noteAIFrame(performance.now()-lastRemoteSent);
      };
      picture.src=msg.image;
    };
  }catch(err){setBadge('REMOTE ERROR','error');showOverlay('Could not open WebSocket',String(err),true);}
}
$('connect').addEventListener('click',connectRemote);
$('engine').addEventListener('change',e=>{
  currentEngine=e.target.value;
  $('remote-settings').hidden=currentEngine!=='remote';
  hasAIFrame=false;inferMs=null;measuredAiFps=0;aiCount=0;lastAIFrame=0;
  $('ai-latency').textContent='LATENCY —';
  if(currentEngine==='onnx')startNeural();
  else if(currentEngine==='remote'){stopWorker();setBadge('NOT CONNECTED','pending');setModeLabel('DIFFUSION / EXTERNAL GPU');showOverlay('Remote mode selected','Enter a WebSocket URL and press CONNECT.');$('truth').textContent='Remote diffusion is not hosted by GitHub Pages. Connect a separate GPU inference service.';}
  else{stopWorker();stopSocket();hideOverlay();setBadge('NOT AI','pending');setModeLabel('PREVIEW / NO INFERENCE');$('model-corner').textContent='RDR_02 · RAW';$('ai-latency').textContent='LATENCY N/A';$('truth').textContent='Raw downsampled Three.js image. This preview is deliberately NOT described as AI.';}
});

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
    fpsCount=0;fpsTick=time;
    $('scene-fps').textContent=sceneFps+' FPS';
  }
  const needsSource=currentEngine==='preview'||$('show-raw').checked||(currentEngine==='onnx'&&workerReady&&!workerBusy&&time-requestAt>=1000/maxRate)||(currentEngine==='remote'&&socket?.readyState===WebSocket.OPEN&&!remotePending&&time-requestAt>=1000/maxRate)||!hasAIFrame;
  if(needsSource){
    getSourceFrame();
    if(currentEngine==='preview'||$('show-raw').checked||(!hasAIFrame&&!workerReady&&!remotePending)){
      rawPreview();
    }
    if(currentEngine==='onnx'&&workerReady&&!workerBusy&&time-requestAt>=1000/maxRate){
      const pixels=captureCtx.getImageData(0,0,224,224).data;
      workerBusy=true;requestAt=time;
      try{worker.postMessage({type:'frame',pixels:pixels.buffer,sent:performance.now()},[pixels.buffer]);}
      catch{workerBusy=false;}
    }
    if(currentEngine==='remote'&&socket?.readyState===WebSocket.OPEN&&!remotePending&&time-requestAt>=1000/maxRate){
      requestAt=time;remotePending=true;lastRemoteSent=performance.now();
      try{
        socket.send(JSON.stringify({type:'frame',image:capture.toDataURL('image/jpeg',.76),prompt:$('prompt').value,frameId:++sequence}));
      }catch(err){remotePending=false;}
    }
  }
}

resetApple();
resize();
setTimeout(startNeural,350);
requestAnimationFrame(animate);