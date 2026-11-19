/* ReAil's genuine neural-image stage.
   ONNX Model Zoo's Fast Neural Style Transfer: 1x3x224x224 RGB floats 0..255.
   Inference happens off the UI thread. This is not diffusion, and does not
   promise video-rate output on commodity CPUs. */
let ortInstance = null;
let session = null;
let activeStyle = null;
const MODEL_DIM = 224;
const PLANE = MODEL_DIM * MODEL_DIM;
const CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const VALID_STYLES = new Set(['candy','mosaic','rain-princess','udnie']);
function notify(type,extra) { self.postMessage(Object.assign({type},extra||{})); }
async function loadStyle(style) {
  if (!VALID_STYLES.has(style)) throw new Error('Unknown model style: '+style);
  if (!ortInstance) {
    notify('loading',{message:'Starting ONNX WebAssembly engine...'});
    importScripts(CDN+'ort.min.js');
    ortInstance = self.ort;
    ortInstance.env.wasm.wasmPaths = CDN;
    ortInstance.env.wasm.numThreads = 1;
  }
  if (session) { try { await session.release(); } catch {} session = null; }
  notify('loading',{message:'Retrieving '+style+' neural model (~6.7 MB)...'});
  const url = 'https://huggingface.co/onnxmodelzoo/'+style+'-9/resolve/main/'+style+'-9.onnx';
  const response = await fetch(url);
  if (!response.ok) throw new Error('Model download failed: HTTP '+response.status);
  const weights = await response.arrayBuffer();
  notify('loading',{message:'Compiling the neural graph...'});
  session = await ortInstance.InferenceSession.create(weights,{
    executionProviders:['wasm'],
    graphOptimizationLevel:'all'
  });
  activeStyle = style;
  notify('ready',{style});
}
async function processFrame(msg) {
  if (!session) return;
  const bytes = new Uint8ClampedArray(msg.pixels);
  const data = new Float32Array(PLANE*3);
  for(let i=0,j=0;i<PLANE;i++,j+=4) {
    data[i] = bytes[j];
    data[i+PLANE] = bytes[j+1];
    data[i+PLANE*2] = bytes[j+2];
  }
  const tensor = new ortInstance.Tensor('float32',data,[1,3,MODEL_DIM,MODEL_DIM]);
  const inputName = session.inputNames[0] || 'input1';
  const feeds = {};
  feeds[inputName]=tensor;
  const started = performance.now();
  const result = await session.run(feeds);
  const out = result[session.outputNames[0] || 'output1'].data;
  const rgba = new Uint8ClampedArray(PLANE*4);
  for(let i=0,j=0;i<PLANE;i++,j+=4){
    rgba[j] = Math.min(255,Math.max(0,out[i]));
    rgba[j+1] = Math.min(255,Math.max(0,out[i+PLANE]));
    rgba[j+2] = Math.min(255,Math.max(0,out[i+PLANE*2]));
    rgba[j+3] = 255;
  }
  notify('frame',{pixels:rgba.buffer,sent:msg.sent,computeMs:performance.now()-started});
}
let busy = false;
self.onmessage=async({data:msg})=>{
  if(busy)return;
  busy=true;
  try{
    if(msg.type==='init')await loadStyle(msg.style);
    if(msg.type==='frame'){
      const prior = session;
      if(!prior)return;
      const bytes = new Uint8ClampedArray(msg.pixels);
      if(bytes.length!==PLANE*4)throw new Error('Unexpected source frame dimensions');
      await processFrame(msg);
    }
  }catch(e){notify('error',{message:e?.message||String(e)});}
  finally{busy=false;}
};