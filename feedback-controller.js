// ReAil closed-loop feedback controller.
// Sensor and control logic are independent from Three.js and SDXL so they can
// be tested without a GPU. All values are bounded; no additional AI inference.

export const FEEDBACK_PRESETS = Object.freeze({
  stable:   {adhesion: 90, mutation: 15, geometryLock: 95, appearanceMemory: 85},
  balanced: {adhesion: 80, mutation: 35, geometryLock: 85, appearanceMemory: 65},
  mutate:   {adhesion: 50, mutation: 85, geometryLock: 45, appearanceMemory: 35}
});

export const clamp01 = value => Math.max(0, Math.min(1, value));

// The source geometry depth pass writes R/G as 16-bit window depth,
// B = 255 for geometry coverage, and B = 0 for the background.
// readRenderTargetPixels has a bottom-left origin (matching projection UVs).
export function unpackDepth(pixels, offset) {
  return (pixels[offset] + pixels[offset + 1] / 255) / 255;
}

export function depthConfidence(pixels, size, u, v, ndcZ) {
  if (!pixels) return 1; // Graceful fallback if the depth capture is unsupported.
  if (u < 0 || u > 1 || v < 0 || v > 1 || ndcZ < -1 || ndcZ > 1) return 0;
  const x = Math.min(size - 1, Math.max(0, Math.floor(u * size)));
  const y = Math.min(size - 1, Math.max(0, Math.floor(v * size)));
  const expectedDepth = ndcZ * 0.5 + 0.5;
  let bestDelta = Number.POSITIVE_INFINITY;
  let visibleNeighbors = 0;

  // 3×3 sampling prevents missing an edge simply because the low-res depth
  // map's nearest sample falls on the black side of a silhouette.
  for (let oy = -1; oy <= 1; oy++) {
    const sy = y + oy;
    if (sy < 0 || sy >= size) continue;
    for (let ox = -1; ox <= 1; ox++) {
      const sx = x + ox;
      if (sx < 0 || sx >= size) continue;
      const index = (sy * size + sx) * 4;
      if (pixels[index + 2] < 128) continue;
      visibleNeighbors++;
      const d = unpackDepth(pixels, index);
      const difference = Math.abs(d - expectedDepth);
      if (difference < bestDelta) bestDelta = difference;
    }
  }

  if (visibleNeighbors === 0 || bestDelta > 0.022) return 0;
  // Prefer accurately visible regions; reduce confidence at silhouette edges.
  const depthWeight = 1 - clamp01(bestDelta / 0.024);
  return depthWeight * (0.60 + 0.40 * Math.min(1, visibleNeighbors / 6));
}

// Sparse RGB image at the same resolution as geometryMaskSize, top-left origin.
// Return a coarse measurable silhouette error, not an AI semantic judgement.
// It cannot tell whether the *identity* of a generated object is correct.
export function measureSilhouetteDrift(rgbData, depthPixels, size) {
  if (!rgbData || !depthPixels || size < 2) return {drift:0, leakage:0, missing:0, coverage:0};

  // Infer background appearance from the rendered AI image's four corners.
  // This is more robust to global brightness / palette changes than assuming
  // the model always returns pure black outside the object.
  const cornerCoords = [[0,0],[size-1,0],[0,size-1],[size-1,size-1]];
  const background = [0,0,0];
  for (const [x,y] of cornerCoords) {
    const o = (y*size+x)*4;
    background[0] += rgbData[o];
    background[1] += rgbData[o+1];
    background[2] += rgbData[o+2];
  }
  for(let c=0;c<3;c++) background[c] /= cornerCoords.length;

  let outside = 0, inside = 0, leakage = 0, missing = 0;

  for (let y=0;y<size;y++) {
    for (let x=0;x<size;x++) {
      const offset = (y*size+x)*4;
      const depthOffset = ((size - 1 - y) * size + x)*4;
      const expected = depthPixels[depthOffset + 2] >= 128;
      const difference = (
        Math.abs(rgbData[offset] - background[0]) +
        Math.abs(rgbData[offset+1] - background[1]) +
        Math.abs(rgbData[offset+2] - background[2])
      ) / 3;
      // Treat subtle shadow variation and compression noise as background.
      const foreground = difference > 29;
      if (expected) {
        inside++;
        if (!foreground) missing++;
      } else {
        outside++;
        if (foreground) leakage++;
      }
    }
  }

  const spill = outside ? leakage / outside : 0;
  const loss = inside ? missing / inside : 0;
  return {
    drift:clamp01(0.78 * spill + 0.22 * loss),
    leakage:spill,
    missing:loss,
    coverage:inside/(size*size)
  };
}

export class ClosedLoopController {
  constructor() { this.reset(); }

  reset() {
    this.gain = 1;
    this.drift = 0;
    this.leakage = 0;
    this.missing = 0;
    this.coverage = 0;
    this.integral = 0;
    this.previousError = 0;
    this.hasSample = false;
  }

  observe(imageData, depthPixels, size, controls) {
    const sensor = measureSilhouetteDrift(imageData, depthPixels, size);
    const lock = clamp01((controls.geometryLock ?? 85)/100);
    const mutation = clamp01((controls.mutation ?? 35)/100);

    // Higher lock asks for less drift. Mutation deliberately permits more.
    const target = 0.045 + mutation * 0.14 + (1-lock) * 0.09;
    this.drift = this.hasSample ? this.drift * 0.72 + sensor.drift * 0.28 : sensor.drift;
    this.leakage = sensor.leakage;
    this.missing = sensor.missing;
    this.coverage = sensor.coverage;

    const error = target - this.drift;
    this.integral = Math.max(-0.35, Math.min(0.35, this.integral + error * 0.12));
    const derivative = this.hasSample ? error - this.previousError : 0;
    const candidate = 1 + 1.5*error + 0.35*this.integral + 0.12*derivative;
    const bounded = Math.max(0.25, Math.min(1.13, candidate));
    this.gain = this.hasSample ? this.gain*0.68 + bounded*0.32 : bounded;
    this.previousError = error;
    this.hasSample = true;
    return {gain:this.gain, drift:this.drift, target, ...sensor};
  }
}
