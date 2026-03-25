/**
 * Port of sharp/utils/gaussians.py unproject_gaussians + apply_transform.
 *
 * Converts model output (NDC-space Gaussians) to 3D world-space Gaussians
 * suitable for rendering.
 */

export interface Gaussians3D {
  /** (N, 3) world-space positions */
  means: Float32Array;
  /** (N, 3) scale values */
  scales: Float32Array;
  /** (N, 4) quaternions (w, x, y, z) */
  quaternions: Float32Array;
  /** (N, 3) linear RGB colors */
  colors: Float32Array;
  /** (N,) opacity values */
  opacities: Float32Array;
  count: number;
}

/**
 * 4x4 matrix inversion (column-major or row-major — we use row-major here).
 * Simple implementation for a single matrix.
 */
function invert4x4(m: number[]): number[] {
  const inv = new Array(16);
  inv[0] =
    m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
    m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] =
    -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
    m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] =
    m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
    m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] =
    -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
    m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] =
    -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
    m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] =
    m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
    m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] =
    -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
    m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] =
    m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
    m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] =
    m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
    m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] =
    -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
    m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] =
    m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
    m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] =
    -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
    m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] =
    -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
    m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] =
    m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
    m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] =
    -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
    m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] =
    m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
    m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

  const det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (Math.abs(det) < 1e-10) throw new Error("Singular matrix");

  const invDet = 1.0 / det;
  for (let i = 0; i < 16; i++) inv[i] *= invDet;
  return inv;
}

/**
 * Multiply 4x4 matrices A * B (row-major).
 */
function mul4x4(a: number[], b: number[]): number[] {
  const r = new Array(16).fill(0);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      for (let k = 0; k < 4; k++) {
        r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
      }
    }
  }
  return r;
}

/**
 * Build the unprojection matrix: inv(ndc_matrix @ intrinsics @ extrinsics)
 *
 * Following gaussians.py get_unprojection_matrix exactly.
 */
function getUnprojectionMatrix(
  fPx: number,
  imageWidth: number,
  imageHeight: number
): number[] {
  // Intrinsics (4x4, row-major) — matching predict.py's intrinsics_resized
  const intrinsics = [
    fPx, 0, imageWidth / 2, 0,
    0, fPx, imageHeight / 2, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];

  // NDC matrix: converts pixel coords to [-1, 1] range
  const ndc = [
    2.0 / imageWidth, 0, -1, 0,
    0, 2.0 / imageHeight, -1, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];

  // Extrinsics = identity (same as predict.py)
  const extrinsics = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];

  return invert4x4(mul4x4(ndc, mul4x4(intrinsics, extrinsics)));
}

/**
 * Compute rotation matrix from quaternion (w, x, y, z convention).
 * Returns 3x3 as flat array (row-major).
 */
function quatToRotation(w: number, x: number, y: number, z: number): number[] {
  const len = Math.sqrt(w * w + x * x + y * y + z * z);
  const nw = w / len, nx = x / len, ny = y / len, nz = z / len;

  return [
    1 - 2 * (ny * ny + nz * nz), 2 * (nx * ny - nw * nz), 2 * (nx * nz + nw * ny),
    2 * (nx * ny + nw * nz), 1 - 2 * (nx * nx + nz * nz), 2 * (ny * nz - nw * nx),
    2 * (nx * nz - nw * ny), 2 * (ny * nz + nw * nx), 1 - 2 * (nx * nx + ny * ny),
  ];
}

/**
 * Convert linearRGB to sRGB (matching sharp/utils/color_space.py).
 * Public renderers expect sRGB, so we convert during postprocessing.
 */
function linearToSRGB(c: number): number {
  if (c <= 0.0031308) return 12.92 * c;
  return 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
}

/**
 * Unproject NDC Gaussians to world space and prepare for rendering.
 *
 * Port of:
 *   predict.py: predict_image() postprocessing
 *   gaussians.py: unproject_gaussians() + apply_transform()
 */
export function unprojectGaussians(
  meanVectors: Float32Array, // (N, 3) from ONNX
  singularValues: Float32Array, // (N, 3)
  quaternions: Float32Array, // (N, 4)
  colors: Float32Array, // (N, 3) linearRGB
  opacities: Float32Array, // (N,)
  count: number,
  imageWidth: number,
  _imageHeight: number,
  fPx: number
): Gaussians3D {
  // Internal resolution used by the model
  const internalSize = 1536;
  const fPxResized = fPx * (internalSize / imageWidth);

  const unprojMatrix = getUnprojectionMatrix(fPxResized, internalSize, internalSize);

  // Extract 3x3 linear part and 3x1 offset from the unprojection matrix (top 3 rows)
  const R = [
    unprojMatrix[0], unprojMatrix[1], unprojMatrix[2],
    unprojMatrix[4], unprojMatrix[5], unprojMatrix[6],
    unprojMatrix[8], unprojMatrix[9], unprojMatrix[10],
  ];
  const t = [unprojMatrix[3], unprojMatrix[7], unprojMatrix[11]];

  const outMeans = new Float32Array(count * 3);
  const outScales = new Float32Array(count * 3);
  const outQuats = new Float32Array(count * 4);
  const outColors = new Float32Array(count * 3);
  const outOpacities = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const mx = meanVectors[i * 3];
    const my = meanVectors[i * 3 + 1];
    const mz = meanVectors[i * 3 + 2];

    // Transform mean: R @ mean + t
    const wx = R[0] * mx + R[1] * my + R[2] * mz + t[0];
    const wy = R[3] * mx + R[4] * my + R[5] * mz + t[1];
    const wz = R[6] * mx + R[7] * my + R[8] * mz + t[2];

    // SHARP: +X right, +Y down, +Z forward (into scene)
    // Three.js: +X right, +Y up, -Z forward
    outMeans[i * 3] = wx;
    outMeans[i * 3 + 1] = -wy;
    outMeans[i * 3 + 2] = -wz;

    // Transform covariance: R @ (quat_to_rot @ diag(s^2) @ quat_to_rot^T) @ R^T
    // Then decompose back to quaternion + singular values.
    // For the PoC, we apply the linear transform to the scale/rotation:
    const qw = quaternions[i * 4];
    const qx = quaternions[i * 4 + 1];
    const qy = quaternions[i * 4 + 2];
    const qz = quaternions[i * 4 + 3];
    const rot = quatToRotation(qw, qx, qy, qz);

    const sx = singularValues[i * 3];
    const sy = singularValues[i * 3 + 1];
    const sz = singularValues[i * 3 + 2];

    // Covariance = rot @ diag(s^2) @ rot^T
    // Transformed covariance = R @ cov @ R^T
    // Combined rotation = R @ rot, scales stay the same (when R is ~uniform scale)
    // Full correct version: compose cov, transform, SVD decompose.
    // Simplified: R @ rot gives new rotation, scale adjusted by det(R)^(1/3)
    // This is approximate but visually close when R is near-orthogonal.

    // Compute R @ rot (3x3 multiply)
    const newRot = new Array(9);
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        newRot[row * 3 + col] =
          R[row * 3] * rot[col] +
          R[row * 3 + 1] * rot[3 + col] +
          R[row * 3 + 2] * rot[6 + col];
      }
    }

    // Extract quaternion from rotation matrix
    const trace = newRot[0] + newRot[4] + newRot[8];
    let nw: number, nx: number, ny: number, nz: number;
    if (trace > 0) {
      const s = 0.5 / Math.sqrt(trace + 1.0);
      nw = 0.25 / s;
      nx = (newRot[7] - newRot[5]) * s;
      ny = (newRot[2] - newRot[6]) * s;
      nz = (newRot[3] - newRot[1]) * s;
    } else if (newRot[0] > newRot[4] && newRot[0] > newRot[8]) {
      const s = 2.0 * Math.sqrt(1.0 + newRot[0] - newRot[4] - newRot[8]);
      nw = (newRot[7] - newRot[5]) / s;
      nx = 0.25 * s;
      ny = (newRot[1] + newRot[3]) / s;
      nz = (newRot[2] + newRot[6]) / s;
    } else if (newRot[4] > newRot[8]) {
      const s = 2.0 * Math.sqrt(1.0 + newRot[4] - newRot[0] - newRot[8]);
      nw = (newRot[2] - newRot[6]) / s;
      nx = (newRot[1] + newRot[3]) / s;
      ny = 0.25 * s;
      nz = (newRot[5] + newRot[7]) / s;
    } else {
      const s = 2.0 * Math.sqrt(1.0 + newRot[8] - newRot[0] - newRot[4]);
      nw = (newRot[3] - newRot[1]) / s;
      nx = (newRot[2] + newRot[6]) / s;
      ny = (newRot[5] + newRot[7]) / s;
      nz = 0.25 * s;
    }
    // Apply Y/Z flip to quaternion: negate y and z components
    // This is equivalent to conjugating by the flip matrix diag(1, -1, -1)
    outQuats[i * 4] = nw;
    outQuats[i * 4 + 1] = nx;
    outQuats[i * 4 + 2] = -ny;
    outQuats[i * 4 + 3] = -nz;

    // Scale adjusted by the uniform scale factor of R
    // det(R)^(1/3) approximates the uniform scaling
    const detR =
      R[0] * (R[4] * R[8] - R[5] * R[7]) -
      R[1] * (R[3] * R[8] - R[5] * R[6]) +
      R[2] * (R[3] * R[7] - R[4] * R[6]);
    const scaleFactor = Math.cbrt(Math.abs(detR));
    outScales[i * 3] = sx * scaleFactor;
    outScales[i * 3 + 1] = sy * scaleFactor;
    outScales[i * 3 + 2] = sz * scaleFactor;

    // Convert linearRGB -> sRGB, clamp to [0, 1]
    outColors[i * 3] = Math.max(0, Math.min(1, linearToSRGB(colors[i * 3])));
    outColors[i * 3 + 1] = Math.max(0, Math.min(1, linearToSRGB(colors[i * 3 + 1])));
    outColors[i * 3 + 2] = Math.max(0, Math.min(1, linearToSRGB(colors[i * 3 + 2])));

    outOpacities[i] = opacities[i];
  }

  return {
    means: outMeans,
    scales: outScales,
    quaternions: outQuats,
    colors: outColors,
    opacities: outOpacities,
    count,
  };
}
