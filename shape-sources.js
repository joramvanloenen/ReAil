import * as THREE from 'three';

// Shape sources return the same triangle format ReAil already uses for
// slicing, shader feedback and frame-locked projected texture coordinates.
const grey = (v = 0.52) => new THREE.Color().setRGB(v, v, v);
const vertex = (p, n, color = 0.52) => ({p, n, c: grey(color)});
const face = (a, b, c) => ({v: [a, b, c], mat: 0});

function triangulatedSurface(sample, around = 96, rows = 64) {
  const out = [];
  for (let y = 0; y < rows; y++) {
    const t0 = y / rows, t1 = (y + 1) / rows;
    for (let x = 0; x < around; x++) {
      const a0 = x / around * Math.PI * 2;
      const a1 = (x + 1) / around * Math.PI * 2;
      const a = sample(a0, t0), b = sample(a1, t0);
      const c = sample(a1, t1), d = sample(a0, t1);
      // Outward-facing winding; vertex normals point out of the object too.
      if (y !== 0) out.push(face(a, b, c));
      if (y !== rows - 1) out.push(face(a, c, d));
    }
  }
  return out;
}

export function makeSphereFaces(radius = 1.04, around = 96, rows = 64, color = 0.52) {
  return triangulatedSurface((theta, t) => {
    const phi = t * Math.PI;
    const p = new THREE.Vector3(
      radius * Math.sin(phi) * Math.cos(theta),
      radius * Math.cos(phi),
      radius * Math.sin(phi) * Math.sin(theta)
    );
    const n = p.clone().normalize();
    return vertex(p, n, color);
  }, around, rows);
}

const gaussian = (value, center, spread) =>
  Math.exp(-(((value - center) / spread) ** 2));

// A single clay-like, closed triangulated head. Facial structures deform
// the surface itself rather than attaching decorative objects that cannot
// participate in slicing and AI texture projection.
function headPoint(theta, t) {
  const phi = t * Math.PI;
  const y = Math.cos(phi) * 1.14;
  const s = Math.sin(phi);
  const jaw = 1 - 0.20 * gaussian(y, -0.84, 0.33);
  let x = Math.cos(theta) * s * 0.80 * jaw;
  let z = Math.sin(theta) * s * 0.72;

  const front = Math.max(0, Math.sin(theta));
  const back = Math.max(0, -Math.sin(theta));
  const ax = Math.abs(x);
  // Forehead, temples and cheekbones.
  z += front * (
    0.065 * gaussian(y, 0.66, 0.35) +
    0.09 * gaussian(ax, 0.40, 0.22) * gaussian(y, -0.05, 0.29)
  );
  // Eye sockets and heavy brow ridge.
  z += front * (
    -0.16 * gaussian(ax, 0.28, 0.16) * gaussian(y, 0.25, 0.17) +
    0.085 * gaussian(ax, 0.28, 0.20) * gaussian(y, 0.45, 0.10)
  );
  // Nose bridge, tip and projecting nostril wings.
  z += front * (
    0.17 * gaussian(x, 0, 0.14) * gaussian(y, 0.25, 0.34) +
    0.37 * gaussian(x, 0, 0.17) * gaussian(y, -0.12, 0.17) +
    0.12 * gaussian(ax, 0.15, 0.075) * gaussian(y, -0.23, 0.09) -
    0.065 * gaussian(x, 0, 0.20) * gaussian(y, -0.30, 0.035)
  );
  // Lips, mouth indentation, projecting chin and angular jaw.
  z += front * (
    0.09 * gaussian(x, 0, 0.32) * gaussian(y, -0.46, 0.075) -
    0.075 * gaussian(x, 0, 0.33) * gaussian(y, -0.53, 0.040) +
    0.13 * gaussian(x, 0, 0.37) * gaussian(y, -0.75, 0.18)
  );
  // Ear lobes grow from the same connected head surface.
  const ear = gaussian(ax, 0.78, 0.13) * gaussian(y, -0.025, 0.31);
  x += Math.sign(x) * 0.21 * ear;
  z += 0.03 * ear;
  z -= back * 0.03 * gaussian(y, 0, 0.60);
  return new THREE.Vector3(x, y, z);
}

export function makeHeadFaces() {
  return triangulatedSurface((theta, t) => {
    const p = headPoint(theta, t);
    const eps = 0.00025;
    const da = headPoint(theta + eps, t).sub(headPoint(theta - eps, t));
    const dt = headPoint(theta, Math.min(0.99999, t + eps))
      .sub(headPoint(theta, Math.max(0.00001, t - eps)));
    const n = da.cross(dt).normalize();
    if (n.lengthSq() < 0.1) n.copy(p).normalize();
    // Ensure normals point outward on the facial features.
    if (n.dot(p) < 0) n.negate();
    const tone = 0.49 + 0.012 * Math.sin(theta * 17 + t * 31);
    return vertex(p, n, tone);
  }, 112, 72);
}

export function initialBlobNodes() {
  return [
    {position: new THREE.Vector3(-0.43, 0.22, 0.00), radius: 0.67},
    {position: new THREE.Vector3(0.39, 0.20, 0.02), radius: 0.65},
    {position: new THREE.Vector3(0.10, -0.46, 0.22), radius: 0.61},
    {position: new THREE.Vector3(-0.06, 0.37, -0.34), radius: 0.54}
  ];
}

// A 3D scalar field extracted with marching tetrahedra. It creates a
// continuous fused surface, unlike intersecting spheres.
export function makeBlobFaces(nodes, resolution = 26) {
  if (!nodes.length) return [];
  const N = Math.max(14, Math.min(34, resolution));
  const half = Math.max(1.65, ...nodes.map(node => (
    Math.max(
      Math.abs(node.position.x),
      Math.abs(node.position.y),
      Math.abs(node.position.z)
    ) + node.radius * 1.5
  )));
  const step = half * 2 / N;
  const side = N + 1;
  const count = side ** 3;
  const samples = new Float32Array(count);
  const index = (x, y, z) => (z * side + y) * side + x;
  const level = 0.56;

  const data = nodes.map(node => ({
    x: node.position.x, y: node.position.y, z: node.position.z,
    invR2: 1 / (node.radius * node.radius)
  }));

  function field(x, y, z) {
    let sum = 0;
    for (const node of data) {
      const dx = x - node.x, dy = y - node.y, dz = z - node.z;
      sum += Math.exp(-(dx * dx + dy * dy + dz * dz) * node.invR2);
    }
    return sum;
  }

  function normalAt(p) {
    let nx = 0, ny = 0, nz = 0;
    for (const node of data) {
      const dx = p.x - node.x, dy = p.y - node.y, dz = p.z - node.z;
      const k = Math.exp(-(dx * dx + dy * dy + dz * dz) * node.invR2) *
        (2 * node.invR2);
      nx += dx * k;
      ny += dy * k;
      nz += dz * k;
    }
    const n = new THREE.Vector3(nx, ny, nz);
    return n.lengthSq() < 1e-10 ? new THREE.Vector3(0, 1, 0) : n.normalize();
  }

  for (let z = 0; z <= N; z++) {
    const pz = -half + z * step;
    for (let y = 0; y <= N; y++) {
      const py = -half + y * step;
      for (let x = 0; x <= N; x++) {
        samples[index(x, y, z)] = field(-half + x * step, py, pz);
      }
    }
  }

  const offsets = [
    [0,0,0], [1,0,0], [1,1,0], [0,1,0],
    [0,0,1], [1,0,1], [1,1,1], [0,1,1]
  ];
  const tetrahedra = [
    [0,5,1,6], [0,1,2,6], [0,2,3,6],
    [0,3,7,6], [0,7,4,6], [0,4,5,6]
  ];
  const faces = [];
  const cubePositions = new Array(8);
  const cubeValues = new Array(8);

  function interpolate(ia, ib, points, values) {
    const v0 = values[ia], v1 = values[ib];
    const t = Math.max(0, Math.min(1, (level - v0) / (v1 - v0 || 1e-8)));
    return points[ia].clone().lerp(points[ib], t);
  }

  function emitTriangle(a, b, c) {
    const edge1 = b.clone().sub(a);
    const edge2 = c.clone().sub(a);
    const geometric = edge1.cross(edge2);
    if (geometric.lengthSq() < 1e-12) return;
    const averageNormal = normalAt(a.clone().add(b).add(c).multiplyScalar(1/3));
    if (geometric.dot(averageNormal) < 0) [b, c] = [c, b];
    faces.push(face(
      vertex(a, normalAt(a), 0.52),
      vertex(b, normalAt(b), 0.52),
      vertex(c, normalAt(c), 0.52)
    ));
  }

  for (let z = 0; z < N; z++) {
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        let allInside = true, allOutside = true;

        for (let i = 0; i < 8; i++) {
          const [ox, oy, oz] = offsets[i];
          cubeValues[i] = samples[index(x + ox, y + oy, z + oz)];
          if (cubeValues[i] < level) allInside = false;
          else allOutside = false;
        }
        if (allInside || allOutside) continue;
        // Only allocate vertex vectors for the small number of cells that
        // actually cross the implicit surface.
        for (let i = 0; i < 8; i++) {
          const [ox, oy, oz] = offsets[i];
          cubePositions[i] = new THREE.Vector3(
            -half + (x + ox) * step,
            -half + (y + oy) * step,
            -half + (z + oz) * step
          );
        }

        for (const tet of tetrahedra) {
          const inside = tet.filter(i => cubeValues[i] >= level);
          const outside = tet.filter(i => cubeValues[i] < level);
          if (!inside.length || !outside.length) continue;

          if (inside.length === 1 || inside.length === 3) {
            const lone = inside.length === 1 ? inside[0] : outside[0];
            const others = inside.length === 1 ? outside : inside;
            emitTriangle(
              interpolate(lone, others[0], cubePositions, cubeValues),
              interpolate(lone, others[1], cubePositions, cubeValues),
              interpolate(lone, others[2], cubePositions, cubeValues)
            );
          } else {
            const a = interpolate(inside[0], outside[0], cubePositions, cubeValues);
            const b = interpolate(inside[0], outside[1], cubePositions, cubeValues);
            const c = interpolate(inside[1], outside[0], cubePositions, cubeValues);
            const d = interpolate(inside[1], outside[1], cubePositions, cubeValues);
            emitTriangle(a, b, c);
            emitTriangle(b, d, c);
          }
        }
      }
    }
  }
  return faces;
}

export function makeParticleFaces() {
  return makeSphereFaces(1, 18, 12, 0.58);
}
