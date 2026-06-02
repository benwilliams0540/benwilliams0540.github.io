const manifestURL = "./prototype-manifest.json";
const canvas = document.querySelector("#splat-canvas");
const statusBar = document.querySelector(".status-bar");
const statusText = document.querySelector("#renderer-status");
const sceneTitle = document.querySelector("#scene-title");
const sceneTabs = document.querySelector("#scene-tabs");
const fallbackLink = document.querySelector("#fallback-link");
const modeToggle = document.querySelector("#mode-toggle");
const rotationToggle = document.querySelector("#rotation-toggle");
const frameButton = document.querySelector("#frame-button");
const diagnosticModeSelect = document.querySelector("#diagnostic-mode");
const fovControl = document.querySelector("#fov-control");
const fovValue = document.querySelector("#fov-value");
const exposureControl = document.querySelector("#exposure-control");
const exposureValue = document.querySelector("#exposure-value");
const pixelScaleControl = document.querySelector("#pixel-scale-control");
const pixelScaleValue = document.querySelector("#pixel-scale-value");
const shBandControl = document.querySelector("#sh-band-control");
const toneMappingControl = document.querySelector("#tone-mapping-control");
const pressedNavigationKeys = new Set();
let shiftIsPressed = false;
const SH_C0 = 0.28209479177387814;
const COVARIANCE_SH0_STRIDE_FLOATS = 14;
const SH_COEFFICIENT_COUNT = 15;
const BASE_SPLAT_STRIDE_FLOATS = 20;
const SH_PALETTE_STRIDE_FLOATS = SH_COEFFICIENT_COUNT * 3;
const EXPANDED_SH3_STRIDE_FLOATS = COVARIANCE_SH0_STRIDE_FLOATS + SH_PALETTE_STRIDE_FLOATS;
const SPLAT_STRIDE_FLOATS = BASE_SPLAT_STRIDE_FLOATS;
const diagnosticModeLabels = new Map([
  [0, "normal"],
  [1, "opaque depth"],
  [2, "no sort"],
  [3, "SH0 only"],
  [4, "radius heat"],
  [5, "cull debug"],
  [6, "alpha heat"],
]);
const toneMappingModes = new Map([
  ["linear", 0],
  ["reinhard", 1],
  ["aces", 2],
]);
const shCoefficientsByBand = [0, 3, 8, 15];
const defaultRenderSettings = {
  covarianceBlur: 0.3,
  exposure: 1,
  fovDegrees: 60,
  pixelScale: 1,
  shBandCount: 3,
  toneMapping: "aces",
};
const navigationKeyCodes = new Set([
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "ArrowUp",
  "KeyA",
  "KeyD",
  "KeyE",
  "KeyQ",
  "KeyS",
  "KeyW",
  "Space",
]);

const quad = new Float32Array([
  -1, -1,
   1, -1,
  -1,  1,
  -1,  1,
   1, -1,
   1,  1,
]);

const shader = `
const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2_0: f32 = 1.0925484305920792;
const SH_C2_1: f32 = -1.0925484305920792;
const SH_C2_2: f32 = 0.31539156525252005;
const SH_C2_3: f32 = -1.0925484305920792;
const SH_C2_4: f32 = 0.5462742152960396;
const SH_C3_0: f32 = -0.5900435899266435;
const SH_C3_1: f32 = 2.890611442640554;
const SH_C3_2: f32 = -0.4570457994644658;
const SH_C3_3: f32 = 0.3731763325901154;
const SH_C3_4: f32 = -0.4570457994644658;
const SH_C3_5: f32 = 1.445305721320277;
const SH_C3_6: f32 = -0.5900435899266435;
const K_BOUNDS_RADIUS: f32 = 3.0;
const K_BOUNDS_RADIUS_SQUARED: f32 = K_BOUNDS_RADIUS * K_BOUNDS_RADIUS;
const SPLAT_STRIDE: u32 = 20u;
const SH_PALETTE_STRIDE: u32 = 45u;

struct CameraUniform {
  viewProjection: mat4x4<f32>,
  view: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  viewport: vec4<f32>,
  fov: vec4<f32>,
  render: vec4<f32>,
};

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> splats: array<f32>;
@group(0) @binding(2) var<storage, read> corners: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> shPalette: array<f32>;
@group(0) @binding(4) var<storage, read> sortedIndices: array<u32>;

struct Axes {
  axis1: vec2<f32>,
  axis2: vec2<f32>,
};

fn splatFloat(splatIndex: u32, component: u32) -> f32 {
  return splats[(splatIndex * SPLAT_STRIDE) + component];
}

fn rotateVectorByQuaternion(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let u = q.xyz;
  let s = q.w;
  let t = 2.0 * cross(u, v);
  return v + (s * t) + cross(u, t);
}

fn shCoefficient(splatIndex: u32, coefficientIndex: u32) -> vec3<f32> {
  let coefficientCount = u32(camera.render.w + 0.5);
  if (coefficientIndex >= coefficientCount) {
    return vec3<f32>(0.0);
  }
  let paletteIndexFloat = splatFloat(splatIndex, 3u);
  if (paletteIndexFloat < 0.0) {
    return vec3<f32>(0.0);
  }
  let paletteIndex = u32(paletteIndexFloat);
  let offset = (paletteIndex * SH_PALETTE_STRIDE) + (coefficientIndex * 3u);
  return vec3<f32>(shPalette[offset], shPalette[offset + 1u], shPalette[offset + 2u]);
}

fn evaluateSH(splatIndex: u32, direction: vec3<f32>) -> vec3<f32> {
  let sh0 = shCoefficient(splatIndex, 0u);
  let sh1 = shCoefficient(splatIndex, 1u);
  let sh2 = shCoefficient(splatIndex, 2u);
  let sh3 = shCoefficient(splatIndex, 3u);
  let sh4 = shCoefficient(splatIndex, 4u);
  let sh5 = shCoefficient(splatIndex, 5u);
  let sh6 = shCoefficient(splatIndex, 6u);
  let sh7 = shCoefficient(splatIndex, 7u);
  let sh8 = shCoefficient(splatIndex, 8u);
  let sh9 = shCoefficient(splatIndex, 9u);
  let sh10 = shCoefficient(splatIndex, 10u);
  let sh11 = shCoefficient(splatIndex, 11u);
  let sh12 = shCoefficient(splatIndex, 12u);
  let sh13 = shCoefficient(splatIndex, 13u);
  let sh14 = shCoefficient(splatIndex, 14u);

  let x = direction.x;
  let y = direction.y;
  let z = direction.z;
  let xx = x * x;
  let yy = y * y;
  let zz = z * z;
  let xy = x * y;
  let yz = y * z;
  let xz = x * z;

  var result = SH_C1 * ((-sh0 * y) + (sh1 * z) - (sh2 * x));
  result += (
    sh3 * (SH_C2_0 * xy)
    + sh4 * (SH_C2_1 * yz)
    + sh5 * (SH_C2_2 * ((2.0 * zz) - xx - yy))
    + sh6 * (SH_C2_3 * xz)
    + sh7 * (SH_C2_4 * (xx - yy))
  );
  result += (
    sh8 * (SH_C3_0 * y * ((3.0 * xx) - yy))
    + sh9 * (SH_C3_1 * xy * z)
    + sh10 * (SH_C3_2 * y * ((4.0 * zz) - xx - yy))
    + sh11 * (SH_C3_3 * z * ((2.0 * zz) - (3.0 * xx) - (3.0 * yy)))
    + sh12 * (SH_C3_4 * x * ((4.0 * zz) - xx - yy))
    + sh13 * (SH_C3_5 * z * (xx - yy))
    + sh14 * (SH_C3_6 * x * (xx - (3.0 * yy)))
  );
  return result;
}

fn heatColor(value: f32) -> vec3<f32> {
  let clampedValue = clamp(value, 0.0, 1.0);
  return vec3<f32>(
    clampedValue,
    1.0 - abs((clampedValue * 2.0) - 1.0),
    1.0 - clampedValue
  );
}

fn calcCovariance2D(viewPosInput: vec3<f32>, covA: vec3<f32>, covB: vec3<f32>) -> vec3<f32> {
  var viewPos = viewPosInput;
  let invViewPosZ = 1.0 / viewPos.z;
  let invViewPosZSquared = invViewPosZ * invViewPosZ;

  let limX = 1.3 * camera.fov.x;
  let limY = 1.3 * camera.fov.y;
  viewPos.x = clamp(viewPos.x * invViewPosZ, -limX, limX) * viewPos.z;
  viewPos.y = clamp(viewPos.y * invViewPosZ, -limY, limY) * viewPos.z;

  let focalX = camera.viewport.z;
  let focalY = camera.viewport.w;
  let j = mat3x3<f32>(
    vec3<f32>(focalX * invViewPosZ, 0.0, 0.0),
    vec3<f32>(0.0, focalY * invViewPosZ, 0.0),
    vec3<f32>(-(focalX * viewPos.x) * invViewPosZSquared, -(focalY * viewPos.y) * invViewPosZSquared, 0.0)
  );
  let w = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
  let t = j * w;
  let cov3D = mat3x3<f32>(
    vec3<f32>(covA.x, covA.y, covA.z),
    vec3<f32>(covA.y, covB.x, covB.y),
    vec3<f32>(covA.z, covB.y, covB.z)
  );
  var cov2D = t * cov3D * transpose(t);
  let covarianceBlur = max(camera.render.z, 0.0);
  cov2D[0][0] += covarianceBlur;
  cov2D[1][1] += covarianceBlur;
  return vec3<f32>(cov2D[0][0], cov2D[0][1], cov2D[1][1]);
}

fn decomposeCovariance(cov2D: vec3<f32>) -> Axes {
  let a = cov2D.x;
  let b = cov2D.y;
  let d = cov2D.z;
  let det = (a * d) - (b * b);
  let trace = a + d;
  let mean = 0.5 * trace;
  let distance = max(0.1, sqrt(max(0.0, (mean * mean) - det)));
  let lambda1 = max(mean + distance, 0.0);
  let lambda2 = max(mean - distance, 0.0);

  var eigenvector1: vec2<f32>;
  if (abs(b) < 0.000001) {
    if (a > d) {
      eigenvector1 = vec2<f32>(1.0, 0.0);
    } else {
      eigenvector1 = vec2<f32>(0.0, 1.0);
    }
  } else {
    eigenvector1 = normalize(vec2<f32>(b, d - lambda2));
  }

  let eigenvector2 = vec2<f32>(eigenvector1.y, -eigenvector1.x);
  return Axes(eigenvector1 * sqrt(lambda1), eigenvector2 * sqrt(lambda2));
}

fn splatFragmentAlpha(relativePosition: vec2<f32>, splatAlpha: f32) -> f32 {
  let negativeMagnitudeSquared = -dot(relativePosition, relativePosition);
  if (negativeMagnitudeSquared < -K_BOUNDS_RADIUS_SQUARED) {
    return 0.0;
  }
  return exp(0.5 * negativeMagnitudeSquared) * splatAlpha;
}

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOut {
  let corner = corners[vertexIndex];
  let splatIndex = sortedIndices[instanceIndex];
  let position = vec3<f32>(
    splatFloat(splatIndex, 0u),
    splatFloat(splatIndex, 1u),
    splatFloat(splatIndex, 2u)
  );
  let rawColor = vec3<f32>(
    splatFloat(splatIndex, 4u),
    splatFloat(splatIndex, 5u),
    splatFloat(splatIndex, 6u)
  );
  let alpha = splatFloat(splatIndex, 7u);
  let covA = vec3<f32>(
    splatFloat(splatIndex, 8u),
    splatFloat(splatIndex, 9u),
    splatFloat(splatIndex, 10u)
  );
  let covB = vec3<f32>(
    splatFloat(splatIndex, 11u),
    splatFloat(splatIndex, 12u),
    splatFloat(splatIndex, 13u)
  );

  var out: VertexOut;
  out.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
  out.local = vec2<f32>(0.0);
  out.color = vec4<f32>(0.0);

  let viewPosition4 = camera.view * vec4<f32>(position, 1.0);
  let viewPosition = viewPosition4.xyz;
  if (viewPosition.z >= 0.0) {
    return out;
  }

  let cov2D = calcCovariance2D(viewPosition, covA, covB);
  let axes = decomposeCovariance(cov2D);
  let projectedCenter = camera.viewProjection * vec4<f32>(position, 1.0);
  let screenSize = camera.viewport.xy;
  let projectedClipExtent = (abs(axes.axis1) + abs(axes.axis2))
    * 2.0
    * K_BOUNDS_RADIUS
    / screenSize
    * projectedCenter.w;
  let bounds = 1.2 * projectedCenter.w;
  let centerOnlyScreenCull = projectedCenter.x < -bounds
    || projectedCenter.x > bounds
    || projectedCenter.y < -bounds
    || projectedCenter.y > bounds;

  if (projectedCenter.z < 0.0
    || projectedCenter.z > projectedCenter.w
    || projectedCenter.x + projectedClipExtent.x < -bounds
    || projectedCenter.x - projectedClipExtent.x > bounds
    || projectedCenter.y + projectedClipExtent.y < -bounds
    || projectedCenter.y - projectedClipExtent.y > bounds) {
    return out;
  }

  let projectedScreenDelta = ((corner.x * axes.axis1) + (corner.y * axes.axis2))
    * 2.0
    * K_BOUNDS_RADIUS
    / screenSize;
  let diagnosticMode = u32(camera.fov.z + 0.5);
  let sh0Color = max(vec3<f32>(0.0), vec3<f32>(0.5) + (rawColor * SH_C0));
  let viewDirectionDelta = camera.cameraPosition.xyz - position;
  let worldViewDirection = viewDirectionDelta / max(length(viewDirectionDelta), 0.000001);
  let rotation = vec4<f32>(
    splatFloat(splatIndex, 14u),
    splatFloat(splatIndex, 15u),
    splatFloat(splatIndex, 16u),
    splatFloat(splatIndex, 17u)
  );
  let localViewDirection = normalize(rotateVectorByQuaternion(vec4<f32>(-rotation.xyz, rotation.w), worldViewDirection));
  var viewColor = sh0Color;
  if (diagnosticMode != 3u) {
    viewColor = max(vec3<f32>(0.0), sh0Color + evaluateSH(splatIndex, localViewDirection));
  }
  var vertexColor = vec4<f32>(viewColor, alpha);

  if (diagnosticMode == 1u) {
    let depth = clamp(projectedCenter.z / max(projectedCenter.w, 0.0001), 0.0, 1.0);
    vertexColor = vec4<f32>(vec3<f32>(depth), 1.0);
  } else if (diagnosticMode == 4u) {
    let radiusPixels = max(length(axes.axis1), length(axes.axis2)) * K_BOUNDS_RADIUS;
    let heat = clamp(log2(max(radiusPixels, 1.0)) / 10.0, 0.0, 1.0);
    vertexColor = vec4<f32>(heatColor(heat), 0.85);
  } else if (diagnosticMode == 5u) {
    if (centerOnlyScreenCull) {
      vertexColor = vec4<f32>(1.0, 0.1, 0.05, 0.9);
    } else {
      vertexColor = vec4<f32>(0.05, 0.35, 1.0, 0.45);
    }
  } else if (diagnosticMode == 6u) {
    vertexColor = vec4<f32>(heatColor(alpha), 0.85);
  }

  let clipX = projectedCenter.x + (projectedScreenDelta.x * projectedCenter.w);
  let clipY = projectedCenter.y + (projectedScreenDelta.y * projectedCenter.w);
  out.position = vec4<f32>(clipX, -clipY, projectedCenter.z, projectedCenter.w);
  out.local = K_BOUNDS_RADIUS * corner;
  out.color = vertexColor;
  return out;
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4<f32> {
  let diagnosticMode = u32(camera.fov.z + 0.5);
  if (diagnosticMode == 1u) {
    return vec4<f32>(in.color.rgb, 1.0);
  }
  let alpha = clamp(splatFragmentAlpha(in.local, in.color.a), 0.0, 1.0);
  let exposure = max(camera.render.x, 0.0);
  let toneMode = u32(camera.render.y + 0.5);
  var color = max(vec3<f32>(0.0), in.color.rgb * exposure);
  if (toneMode == 1u) {
    color = color / (vec3<f32>(1.0) + color);
  } else if (toneMode == 2u) {
    color = clamp((color * (2.51 * color + vec3<f32>(0.03))) / (color * (2.43 * color + vec3<f32>(0.59)) + vec3<f32>(0.14)), vec3<f32>(0.0), vec3<f32>(1.0));
  }
  return vec4<f32>(color * alpha, alpha);
}
`;

const countingSortShader = `
const SPLAT_STRIDE: u32 = 20u;

struct SortUniform {
  eye: vec4<f32>,
  forward: vec4<f32>,
  range: vec4<f32>,
  counts: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> splats: array<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> binOffsets: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(5) var<uniform> sortUniform: SortUniform;

fn splatFloat(splatIndex: u32, component: u32) -> f32 {
  return splats[(splatIndex * SPLAT_STRIDE) + component];
}

@compute @workgroup_size(256)
fn resetBins(@builtin(global_invocation_id) globalID: vec3<u32>) {
  let bin = globalID.x;
  if (bin >= sortUniform.counts.y) {
    return;
  }
  atomicStore(&histogram[bin], 0u);
  atomicStore(&binOffsets[bin], 0u);
}

@compute @workgroup_size(256)
fn buildHistogram(@builtin(global_invocation_id) globalID: vec3<u32>) {
  let splatIndex = globalID.x;
  if (splatIndex >= sortUniform.counts.x) {
    return;
  }

  let position = vec3<f32>(
    splatFloat(splatIndex, 0u),
    splatFloat(splatIndex, 1u),
    splatFloat(splatIndex, 2u)
  );
  let depth = dot(position - sortUniform.eye.xyz, sortUniform.forward.xyz);
  let normalizedDepth = (depth - sortUniform.range.x) * sortUniform.range.y;
  let clampedDepth = clamp(normalizedDepth, 0.0, f32(sortUniform.counts.y - 1u));
  let forwardBin = u32(clampedDepth);
  let bin = (sortUniform.counts.y - 1u) - forwardBin;
  bins[splatIndex] = bin;
  atomicAdd(&histogram[bin], 1u);
}

@compute @workgroup_size(1)
fn prefixHistogram(@builtin(global_invocation_id) globalID: vec3<u32>) {
  if (globalID.x != 0u) {
    return;
  }

  var sum = 0u;
  for (var bin = 0u; bin < sortUniform.counts.y; bin += 1u) {
    let count = atomicLoad(&histogram[bin]);
    atomicStore(&binOffsets[bin], sum);
    sum += count;
  }
}

@compute @workgroup_size(256)
fn scatterIndices(@builtin(global_invocation_id) globalID: vec3<u32>) {
  let splatIndex = globalID.x;
  if (splatIndex >= sortUniform.counts.x) {
    return;
  }

  let bin = bins[splatIndex];
  let position = atomicAdd(&binOffsets[bin], 1u);
  sortedIndices[position] = splatIndex;
}
`;

let manifest;
let scene;
let renderer;

try {
  manifest = await loadManifest();
  scene = selectInitialScene(manifest);
  createTabs(manifest.scenes);
  renderer = await createRenderer(canvas, scene);
  startInteraction(canvas);
  await setScene(scene);
  renderer.start();
} catch (error) {
  setStatus(error.message || String(error), "error");
}

async function loadManifest() {
  const response = await fetch(manifestURL, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load ${manifestURL}`);
  }
  return response.json();
}

async function loadSceneSplats(scene) {
  if (scene.splatBuffer) {
    const response = await fetch(scene.splatBuffer.url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Could not load ${scene.splatBuffer.url}`);
    }

    const buffer = await response.arrayBuffer();
    const sourceStride = scene.splatBuffer.strideFloats ?? SPLAT_STRIDE_FLOATS;
    if (scene.splatBuffer.byteLength && buffer.byteLength !== scene.splatBuffer.byteLength) {
      throw new Error(`Splat buffer size mismatch for ${scene.title}`);
    }
    return binaryBufferToSceneData(buffer, scene.splatBuffer, sourceStride);
  }

  return pointsToFloat32(scene.points ?? []);
}

function binaryBufferToSceneData(buffer, splatBuffer, sourceStride) {
  switch (splatBuffer.format) {
  case "splat-f32-cov-sh-palette-v1": {
    const palette = splatBuffer.shPalette;
    const splatByteLength = palette?.byteOffset ?? buffer.byteLength;
    if (splatByteLength % (sourceStride * 4) !== 0) {
      throw new Error("Invalid palette splat segment size.");
    }
    const splats = new Float32Array(buffer, 0, splatByteLength / 4);
    const shPalette = palette
      ? new Float32Array(buffer, palette.byteOffset, palette.byteLength / 4)
      : new Float32Array();
    return {
      splats,
      shPalette,
      shCoefficientCount: palette?.coefficientCount ?? 0,
      sourceFormat: splatBuffer.format,
    };
  }
  case "splat-f32-cov-sh3-v1":
    if (buffer.byteLength % (sourceStride * 4) !== 0) {
      throw new Error("Invalid expanded SH3 splat buffer size.");
    }
    return expandedSH3ToPaletteData(new Float32Array(buffer), sourceStride);
  default: {
    if (buffer.byteLength % (sourceStride * 4) !== 0) {
      throw new Error("Invalid splat buffer size.");
    }
    const sourceData = new Float32Array(buffer);
    return sourceStride === SPLAT_STRIDE_FLOATS
      ? { splats: sourceData, shPalette: new Float32Array(), shCoefficientCount: 0, sourceFormat: splatBuffer.format }
      : expandLegacySplats(sourceData, sourceStride);
  }
  }
}

function pointsToFloat32(points) {
  const splats = new Float32Array(points.length * SPLAT_STRIDE_FLOATS);
  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const offset = index * SPLAT_STRIDE_FLOATS;
    const radius = point.radius;
    const covariance = radius * radius;
    splats[offset] = point.position.x;
    splats[offset + 1] = point.position.y;
    splats[offset + 2] = point.position.z;
    splats[offset + 3] = 0;
    splats[offset + 4] = displayChannelToSH0(point.color.red);
    splats[offset + 5] = displayChannelToSH0(point.color.green);
    splats[offset + 6] = displayChannelToSH0(point.color.blue);
    splats[offset + 7] = point.color.alpha;
    splats[offset + 8] = covariance;
    splats[offset + 9] = 0;
    splats[offset + 10] = 0;
    splats[offset + 11] = covariance;
    splats[offset + 12] = 0;
    splats[offset + 13] = covariance;
    splats[offset + 14] = 0;
    splats[offset + 15] = 0;
    splats[offset + 16] = 0;
    splats[offset + 17] = 1;
    splats[offset + 18] = 0;
    splats[offset + 19] = 0;
  }
  return { splats, shPalette: new Float32Array(), shCoefficientCount: 0, sourceFormat: "swift-probe" };
}

function expandLegacySplats(sourceData, sourceStride) {
  if (sourceStride !== 8 && sourceStride !== COVARIANCE_SH0_STRIDE_FLOATS) {
    throw new Error(`Unsupported splat buffer stride: ${sourceStride}`);
  }
  const splatCount = sourceData.length / sourceStride;
  const expanded = new Float32Array(splatCount * SPLAT_STRIDE_FLOATS);
  for (let index = 0; index < splatCount; index += 1) {
    const sourceOffset = index * sourceStride;
    const targetOffset = index * SPLAT_STRIDE_FLOATS;
    if (sourceStride === COVARIANCE_SH0_STRIDE_FLOATS) {
      expanded.set(sourceData.subarray(sourceOffset, sourceOffset + sourceStride), targetOffset);
    } else {
      const radius = sourceData[sourceOffset + 3];
      const covariance = radius * radius;
      expanded[targetOffset] = sourceData[sourceOffset];
      expanded[targetOffset + 1] = sourceData[sourceOffset + 1];
      expanded[targetOffset + 2] = sourceData[sourceOffset + 2];
      expanded[targetOffset + 3] = 0;
      expanded[targetOffset + 4] = displayChannelToSH0(sourceData[sourceOffset + 4]);
      expanded[targetOffset + 5] = displayChannelToSH0(sourceData[sourceOffset + 5]);
      expanded[targetOffset + 6] = displayChannelToSH0(sourceData[sourceOffset + 6]);
      expanded[targetOffset + 7] = sourceData[sourceOffset + 7];
      expanded[targetOffset + 8] = covariance;
      expanded[targetOffset + 9] = 0;
      expanded[targetOffset + 10] = 0;
      expanded[targetOffset + 11] = covariance;
      expanded[targetOffset + 12] = 0;
      expanded[targetOffset + 13] = covariance;
    }
    expanded[targetOffset + 14] = 0;
    expanded[targetOffset + 15] = 0;
    expanded[targetOffset + 16] = 0;
    expanded[targetOffset + 17] = 1;
    expanded[targetOffset + 18] = 0;
    expanded[targetOffset + 19] = 0;
  }
  return { splats: expanded, shPalette: new Float32Array(), shCoefficientCount: 0, sourceFormat: "legacy" };
}

function expandedSH3ToPaletteData(sourceData, sourceStride) {
  if (sourceStride !== EXPANDED_SH3_STRIDE_FLOATS) {
    throw new Error(`Unsupported expanded SH3 stride: ${sourceStride}`);
  }
  const splatCount = sourceData.length / sourceStride;
  const splats = new Float32Array(splatCount * SPLAT_STRIDE_FLOATS);
  const shPalette = new Float32Array(splatCount * SH_PALETTE_STRIDE_FLOATS);
  for (let index = 0; index < splatCount; index += 1) {
    const sourceOffset = index * sourceStride;
    const targetOffset = index * SPLAT_STRIDE_FLOATS;
    splats.set(sourceData.subarray(sourceOffset, sourceOffset + COVARIANCE_SH0_STRIDE_FLOATS), targetOffset);
    splats[targetOffset + 3] = index;
    splats[targetOffset + 14] = 0;
    splats[targetOffset + 15] = 0;
    splats[targetOffset + 16] = 0;
    splats[targetOffset + 17] = 1;
    splats[targetOffset + 18] = SH_COEFFICIENT_COUNT;
    splats[targetOffset + 19] = 0;
    shPalette.set(
      sourceData.subarray(sourceOffset + COVARIANCE_SH0_STRIDE_FLOATS, sourceOffset + sourceStride),
      index * SH_PALETTE_STRIDE_FLOATS
    );
  }
  return {
    splats,
    shPalette,
    shCoefficientCount: SH_COEFFICIENT_COUNT,
    sourceFormat: "splat-f32-cov-sh3-v1",
  };
}

function displayChannelToSH0(channel) {
  return (channel - 0.5) / SH_C0;
}

function sequentialIndices(count) {
  const indices = new Uint32Array(count);
  for (let index = 0; index < count; index += 1) {
    indices[index] = index;
  }
  return indices;
}

function selectInitialScene(manifest) {
  const requestedScene = new URLSearchParams(window.location.search).get("scene");
  return manifest.scenes.find((candidate) => candidate.id === requestedScene)
    ?? manifest.scenes[0];
}

function createTabs(scenes) {
  sceneTabs.replaceChildren();
  for (const candidate of scenes) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = candidate.title;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", candidate.id === scene.id ? "true" : "false");
    button.addEventListener("click", async () => {
      const url = new URL(window.location.href);
      url.searchParams.set("scene", candidate.id);
      window.history.replaceState(null, "", url);
      try {
        await setScene(candidate);
      } catch (error) {
        setStatus(error.message || String(error), "error");
      }
    });
    sceneTabs.appendChild(button);
  }
}

async function setScene(nextScene) {
  scene = nextScene;
  sceneTitle.textContent = scene.title;
  fallbackLink.href = scene.fallbackViewer;
  for (const button of sceneTabs.querySelectorAll("button")) {
    button.setAttribute("aria-selected", button.textContent === scene.title ? "true" : "false");
  }
  await renderer?.setScene(scene);
}

function setStatus(message, kind = "ok") {
  statusText.textContent = message;
  statusBar.dataset.kind = kind;
}

async function createRenderer(canvas, initialScene) {
  if (!("gpu" in navigator)) {
    throw new Error("WebGPU is unavailable in this browser. Use the fallback viewer.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU adapter request failed.");
  }

  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });

  const uniformBuffer = device.createBuffer({
    size: 192,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const quadBuffer = device.createBuffer({
    size: quad.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(quadBuffer, 0, quad);

  const module = device.createShaderModule({ code: shader });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vertexMain" },
    fragment: {
      module,
      entryPoint: "fragmentMain",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
  });
  const sortModule = device.createShaderModule({ code: countingSortShader });
  const sortBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const sortPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [sortBindGroupLayout] });
  const sortPipelines = {
    reset: device.createComputePipeline({ layout: sortPipelineLayout, compute: { module: sortModule, entryPoint: "resetBins" } }),
    histogram: device.createComputePipeline({ layout: sortPipelineLayout, compute: { module: sortModule, entryPoint: "buildHistogram" } }),
    prefix: device.createComputePipeline({ layout: sortPipelineLayout, compute: { module: sortModule, entryPoint: "prefixHistogram" } }),
    scatter: device.createComputePipeline({ layout: sortPipelineLayout, compute: { module: sortModule, entryPoint: "scatterIndices" } }),
  };
  const sortBinCount = 4096;

  let splatBuffer;
  let shPaletteBuffer;
  let sortedIndexBuffer;
  let sortBinBuffer;
  let sortHistogramBuffer;
  let sortBinOffsetBuffer;
  let sortUniformBuffer;
  let sortBindGroup;
  let bindGroup;
  let sourceSplatData = new Float32Array();
  let sourceShPalette = new Float32Array();
  let sourceShCoefficientCount = 0;
  let sourceSplatFormat = "unknown";
  let sortedIndexData = new Uint32Array();
  let activeSortSceneID = 0;
  let latestSortRequestID = 0;
  let latestAppliedSortRequestID = 0;
  let cameraRevision = 0;
  let sortedCameraRevision = -1;
  let sortInFlight = false;
  let sortQueued = false;
  let sortState = "sort pending";
  let sortWorker = createSortWorker();
  let renderSettings = normalizedRenderSettings(initialScene.renderSettings);
  let pointerIsActive = false;
  let rotationIsEnabled = false;
  let diagnosticMode = 0;
  let lastSortDuration = 0;
  let framesPerSecond = 0;
  let camera = cameraFromScene(initialScene);
  let frameHandle = 0;
  let lastFrameTime = performance.now();
  let needsSort = true;
  let needsResize = true;

  const resizeObserver = new ResizeObserver(() => {
    needsResize = true;
    requestFrame();
  });
  resizeObserver.observe(canvas);

  function createSortWorker() {
    if (typeof Worker === "undefined") {
      sortState = "main-thread sort";
      return null;
    }

    try {
      const worker = new Worker(new URL("./sort-worker.js", import.meta.url), { type: "module" });
      worker.addEventListener("message", handleSortWorkerMessage);
      worker.addEventListener("error", (event) => {
        console.warn("Splat sort worker failed; falling back to main-thread sorting.", event.message);
        sortState = "main-thread sort";
        sortWorker = null;
        sortInFlight = false;
        sortQueued = false;
        needsSort = true;
        requestFrame();
      });
      return worker;
    } catch (error) {
      console.warn("Splat sort worker could not start; falling back to main-thread sorting.", error);
      sortState = "main-thread sort";
      return null;
    }
  }

  function configureSortWorker() {
    activeSortSceneID += 1;
    latestSortRequestID = 0;
    latestAppliedSortRequestID = 0;
    sortInFlight = false;
    sortQueued = false;

    if (!sortWorker) {
      sortState = "main-thread sort";
      return;
    }

    const workerCopy = new Float32Array(sourceSplatData);
    sortState = "worker ready";
    sortWorker.postMessage({
      type: "setScene",
      sceneID: activeSortSceneID,
      stride: SPLAT_STRIDE_FLOATS,
      splatBuffer: workerCopy.buffer,
    }, [workerCopy.buffer]);
  }

  function handleSortWorkerMessage(event) {
    const message = event.data;
    if (!message || message.sceneID !== activeSortSceneID) return;

    if (message.type === "ready") {
      sortState = "worker ready";
      return;
    }

    if (message.type === "error") {
      console.warn("Splat sort worker returned an error; falling back to main-thread sorting.", message.message);
      sortState = "main-thread sort";
      sortWorker = null;
      sortInFlight = false;
      sortQueued = false;
      needsSort = true;
      requestFrame();
      return;
    }

    if (message.type !== "sorted") return;

    sortInFlight = false;
    if (!splatBuffer || diagnosticMode === 2) return;

    if (message.requestID >= latestAppliedSortRequestID) {
      latestAppliedSortRequestID = message.requestID;
      lastSortDuration = message.sortDuration ?? 0;
      sortedCameraRevision = message.cameraRevision ?? -1;
      device.queue.writeBuffer(sortedIndexBuffer, 0, new Uint32Array(message.sortedIndices));
      sortState = sortedCameraRevision === cameraRevision ? "worker sorted" : "stale worker sort";
    }

    if (sortQueued) {
      sortQueued = false;
      needsSort = true;
    }
    requestFrame();
  }

  async function setScene(nextScene) {
    camera = cameraFromScene(nextScene);
    renderSettings = normalizedRenderSettings(nextScene.renderSettings);
    const sceneData = await loadSceneSplats(nextScene);
    sourceSplatData = sceneData.splats;
    sourceShPalette = sceneData.shPalette;
    sourceShCoefficientCount = sceneData.shCoefficientCount;
    sourceSplatFormat = sceneData.sourceFormat;
    sortedIndexData = sequentialIndices(sourceSplatData.length / SPLAT_STRIDE_FLOATS);
    cameraRevision = 0;
    sortedCameraRevision = -1;
    lastSortDuration = 0;
    splatBuffer = device.createBuffer({
      size: sourceSplatData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(splatBuffer, 0, sourceSplatData);
    shPaletteBuffer = device.createBuffer({
      size: Math.max(sourceShPalette.byteLength, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (sourceShPalette.byteLength > 0) {
      device.queue.writeBuffer(shPaletteBuffer, 0, sourceShPalette);
    }
    sortedIndexBuffer = device.createBuffer({
      size: Math.max(sortedIndexData.byteLength, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(sortedIndexBuffer, 0, sortedIndexData);
    sortBinBuffer = device.createBuffer({
      size: Math.max(sortedIndexData.byteLength, 4),
      usage: GPUBufferUsage.STORAGE,
    });
    sortHistogramBuffer = device.createBuffer({
      size: sortBinCount * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    sortBinOffsetBuffer = device.createBuffer({
      size: sortBinCount * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    sortUniformBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    sortBindGroup = device.createBindGroup({
      layout: sortBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: splatBuffer } },
        { binding: 1, resource: { buffer: sortBinBuffer } },
        { binding: 2, resource: { buffer: sortHistogramBuffer } },
        { binding: 3, resource: { buffer: sortBinOffsetBuffer } },
        { binding: 4, resource: { buffer: sortedIndexBuffer } },
        { binding: 5, resource: { buffer: sortUniformBuffer } },
      ],
    });
    bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: splatBuffer } },
        { binding: 2, resource: { buffer: quadBuffer } },
        { binding: 3, resource: { buffer: shPaletteBuffer } },
        { binding: 4, resource: { buffer: sortedIndexBuffer } },
      ],
    });
    configureSortWorker();
    needsSort = true;
    updateControls();
    setStatus(statusForScene(nextScene, sortedIndexData.length, diagnosticMode, framesPerSecond, lastSortDuration, sortState, sourceSplatFormat, renderSettings));
    requestFrame();
  }

  function start() {
    requestFrame();
  }

  function requestFrame() {
    if (!frameHandle) {
      frameHandle = requestAnimationFrame(render);
    }
  }

  function render() {
    frameHandle = 0;
    if (!splatBuffer || !shPaletteBuffer || !sortedIndexBuffer || !bindGroup) return;

    const now = performance.now();
    const elapsed = Math.min(Math.max((now - lastFrameTime) / 1000, 0), 1 / 15);
    lastFrameTime = now;
    framesPerSecond = elapsed > 0 ? (framesPerSecond * 0.85) + ((1 / elapsed) * 0.15) : framesPerSecond;

    if (navigateByKeyboard(elapsed)) {
      markCameraChanged();
      requestFrame();
    } else if (rotationIsEnabled && !pointerIsActive) {
      camera.yaw -= elapsed * 3 * Math.PI / 180;
      markCameraChanged();
      requestFrame();
    }

    if (needsResize) {
      resizeCanvas(canvas, renderSettings.pixelScale);
      needsResize = false;
    }
    if (needsSort) {
      requestSortedSplats();
      needsSort = false;
    }

    const aspect = canvas.width / Math.max(canvas.height, 1);
    const view = cameraView(camera);
    const projection = cameraProjection(camera, aspect, renderSettings.fovDegrees);
    const viewProjection = multiply(projection, view);
    const eye = cameraEye(camera);
    const uniformData = new Float32Array(48);
    const projection00 = projection[0];
    const projection11 = projection[5];
    uniformData.set(viewProjection, 0);
    uniformData.set(view, 16);
    uniformData[32] = eye.x;
    uniformData[33] = eye.y;
    uniformData[34] = eye.z;
    uniformData[35] = 0;
    uniformData[36] = canvas.width;
    uniformData[37] = canvas.height;
    uniformData[38] = canvas.width * projection00 * 0.5;
    uniformData[39] = canvas.height * projection11 * 0.5;
    uniformData[40] = 1 / projection00;
    uniformData[41] = 1 / projection11;
    uniformData[42] = diagnosticMode;
    uniformData[43] = 0;
    uniformData[44] = renderSettings.exposure;
    uniformData[45] = toneMappingModes.get(renderSettings.toneMapping) ?? 0;
    uniformData[46] = renderSettings.covarianceBlur;
    uniformData[47] = activeShCoefficientCount();
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.02, g: 0.025, b: 0.035, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, sortedIndexData.length);
    pass.end();
    device.queue.submit([encoder.finish()]);

    if (sortState === "worker sorted" && sortedCameraRevision !== cameraRevision) {
      sortState = "stale worker sort";
    }
    setStatus(statusForScene(scene, sortedIndexData.length, diagnosticMode, framesPerSecond, lastSortDuration, sortState, sourceSplatFormat, renderSettings));
  }

  function requestSortedSplats() {
    if (!sourceSplatData.length || !sortedIndexBuffer) return;

    if (diagnosticMode === 2) {
      sortedIndexData = sequentialIndices(sourceSplatData.length / SPLAT_STRIDE_FLOATS);
      device.queue.writeBuffer(sortedIndexBuffer, 0, sortedIndexData);
      lastSortDuration = 0;
      sortedCameraRevision = cameraRevision;
      sortState = "source order";
      return;
    }

    if (writeSortedIndicesWithGPU()) return;

    if (!sortWorker) {
      writeSortedIndicesOnMainThread();
      return;
    }

    if (sortInFlight) {
      sortQueued = true;
      sortState = "worker sorting";
      return;
    }

    const eye = cameraEye(camera);
    const forward = normalize(subtract(cameraTarget(camera), eye));
    latestSortRequestID += 1;
    sortInFlight = true;
    sortQueued = false;
    sortState = "worker sorting";
    sortWorker.postMessage({
      type: "sort",
      sceneID: activeSortSceneID,
      requestID: latestSortRequestID,
      cameraRevision,
      eye,
      forward,
    });
  }

  function writeSortedIndicesWithGPU() {
    if (!sortBindGroup || !sortUniformBuffer) return false;

    try {
      const sortStart = performance.now();
      const eye = cameraEye(camera);
      const forward = normalize(subtract(cameraTarget(camera), eye));
      writeSortUniforms(eye, forward);

      const splatWorkgroups = Math.ceil(sortedIndexData.length / 256);
      const binWorkgroups = Math.ceil(sortBinCount / 256);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, sortBindGroup);
      pass.setPipeline(sortPipelines.reset);
      pass.dispatchWorkgroups(binWorkgroups);
      pass.setPipeline(sortPipelines.histogram);
      pass.dispatchWorkgroups(splatWorkgroups);
      pass.setPipeline(sortPipelines.prefix);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(sortPipelines.scatter);
      pass.dispatchWorkgroups(splatWorkgroups);
      pass.end();
      device.queue.submit([encoder.finish()]);

      lastSortDuration = performance.now() - sortStart;
      sortedCameraRevision = cameraRevision;
      sortState = "gpu counting sorted";
      return true;
    } catch (error) {
      console.warn("WebGPU counting sort failed; falling back to worker sorting.", error);
      sortState = sortWorker ? "worker ready" : "main-thread sort";
      return false;
    }
  }

  function writeSortUniforms(eye, forward) {
    const range = sortDepthRange(eye, forward);
    const uniformBytes = new ArrayBuffer(64);
    const floats = new Float32Array(uniformBytes);
    const uints = new Uint32Array(uniformBytes);

    floats[0] = eye.x;
    floats[1] = eye.y;
    floats[2] = eye.z;
    floats[3] = 0;
    floats[4] = forward.x;
    floats[5] = forward.y;
    floats[6] = forward.z;
    floats[7] = 0;
    floats[8] = range.minDepth;
    floats[9] = sortBinCount / range.depthRange;
    floats[10] = range.depthRange;
    floats[11] = 0;
    uints[12] = sortedIndexData.length;
    uints[13] = sortBinCount;
    uints[14] = 0;
    uints[15] = 0;

    device.queue.writeBuffer(sortUniformBuffer, 0, uniformBytes);
  }

  function sortDepthRange(eye, forward) {
    const bounds = scene?.bounds ?? camera.bounds;
    const center = bounds.center;
    const centerDepth = (
      (center.x - eye.x) * forward.x
      + (center.y - eye.y) * forward.y
      + (center.z - eye.z) * forward.z
    );
    const radius = Math.max(bounds.radius ?? 1, 1);
    const depthRange = Math.max(radius * 2.5, 0.001);
    return {
      minDepth: centerDepth - depthRange * 0.5,
      depthRange,
    };
  }

  function writeSortedIndicesOnMainThread() {
    const sortStart = performance.now();
    const eye = cameraEye(camera);
    const forward = normalize(subtract(cameraTarget(camera), eye));
    const order = Array.from(sortedIndexData);
    order.sort((lhs, rhs) => {
      const lhsOffset = lhs * SPLAT_STRIDE_FLOATS;
      const rhsOffset = rhs * SPLAT_STRIDE_FLOATS;
      const lhsDepth = (
        (sourceSplatData[lhsOffset] - eye.x) * forward.x
        + (sourceSplatData[lhsOffset + 1] - eye.y) * forward.y
        + (sourceSplatData[lhsOffset + 2] - eye.z) * forward.z
      );
      const rhsDepth = (
        (sourceSplatData[rhsOffset] - eye.x) * forward.x
        + (sourceSplatData[rhsOffset + 1] - eye.y) * forward.y
        + (sourceSplatData[rhsOffset + 2] - eye.z) * forward.z
      );
      return rhsDepth - lhsDepth;
    });

    sortedIndexData = Uint32Array.from(order);
    device.queue.writeBuffer(sortedIndexBuffer, 0, sortedIndexData);
    lastSortDuration = performance.now() - sortStart;
    sortedCameraRevision = cameraRevision;
    sortState = "main-thread sorted";
  }

  function markCameraChanged() {
    cameraRevision += 1;
    needsSort = true;
  }

  function orbit(deltaX, deltaY) {
    const settings = camera.settings;
    const eye = cameraEye(camera);
    camera.yaw -= deltaX * settings.orbitSensitivity;
    camera.pitch -= deltaY * settings.orbitSensitivity * pointerPitchDirection(settings);
    camera.pitch = Math.min(Math.max(camera.pitch, -1.45), 1.45);
    if (settings.interactionMode === "fly") {
      const distance = camera.bounds.radius * camera.distance;
      const target = subtract(eye, scale(cameraOrbitDirection(camera), distance));
      camera.targetOffset = scale(subtract(target, camera.bounds.center), 1 / camera.bounds.radius);
    }
    markCameraChanged();
    requestFrame();
  }

  function pan(deltaX, deltaY) {
    const basis = cameraBasis(camera);
    const distance = length(subtract(cameraEye(camera), cameraTarget(camera)));
    const viewportHeight = Math.max(canvas.height, 1);
    const viewportWidth = Math.max(canvas.width, 1);
    const worldPerPixelY = (2 * Math.tan(degreesToRadians(renderSettings.fovDegrees) * 0.5) * distance) / viewportHeight;
    const worldPerPixelX = worldPerPixelY * (viewportWidth / viewportHeight);
    const movement = add(
      scale(basis.right, -deltaX * worldPerPixelX),
      scale(basis.up, deltaY * worldPerPixelY)
    );
    camera.targetOffset = add(
      camera.targetOffset,
      scale(movement, 1 / Math.max(camera.bounds.radius, 0.000001))
    );
    markCameraChanged();
    requestFrame();
  }

  function zoom(deltaY) {
    const factor = Math.exp(-deltaY * 0.0018);
    zoomByFactor(factor);
    markCameraChanged();
    requestFrame();
  }

  function zoomByFactor(factor) {
    const settings = camera.settings;
    if (!Number.isFinite(factor) || factor <= 0) return;

    const adjustedFactor = Math.pow(factor, settings.zoomSensitivity);
    if (!Number.isFinite(adjustedFactor) || adjustedFactor <= 0) return;

    const requestedDistance = camera.distance / adjustedFactor;
    if (requestedDistance < 0.00001 && adjustedFactor > 1) {
      const factorConsumedByDistance = Math.max(camera.distance / 0.00001, 1);
      const residualFactor = Math.max(adjustedFactor / factorConsumedByDistance, 1);
      camera.distance = 0.00001;
      if (settings.dollyPastMinimumIsEnabled) {
        camera.targetOffset = add(
          camera.targetOffset,
          scale(cameraForward(camera), Math.log(residualFactor) * settings.dollySensitivity)
        );
      }
    } else {
      camera.distance = Math.min(Math.max(requestedDistance, 0.00001), 120);
    }
  }

  function navigateByKeyboard(elapsed) {
    const input = keyboardInput();
    if (!input.hasMovementOrRotation || !Number.isFinite(elapsed) || elapsed <= 0) return false;

    const multiplier = Math.max(input.speedMultiplier, 0);
    const movementAmount = elapsed * 0.65 * multiplier;
    const rotationAmount = elapsed * 1.4 * multiplier;
    const forward = cameraForward(camera);
    const right = { x: Math.cos(camera.yaw), y: 0, z: -Math.sin(camera.yaw) };
    const up = { x: 0, y: 1, z: 0 };
    let movement = add(
      add(scale(forward, input.forward), scale(right, input.strafe)),
      scale(up, input.vertical)
    );
    const movementLength = length(movement);
    if (movementLength > 0) {
      movement = scale(movement, movementAmount / movementLength);
      camera.targetOffset = add(camera.targetOffset, movement);
    }

    camera.yaw += input.yaw * rotationAmount;
    camera.pitch += input.pitch * rotationAmount;
    camera.pitch = Math.min(Math.max(camera.pitch, -1.45), 1.45);
    return true;
  }

  function toggleRotation() {
    rotationIsEnabled = !rotationIsEnabled;
    updateControls();
    requestFrame();
  }

  function toggleMode() {
    camera.settings.interactionMode = camera.settings.interactionMode === "fly" ? "orbit" : "fly";
    updateControls();
    requestFrame();
  }

  function frameScene() {
    camera = cameraFromScene(scene);
    markCameraChanged();
    updateControls();
    requestFrame();
  }

  function setDiagnosticMode(nextDiagnosticMode) {
    diagnosticMode = nextDiagnosticMode;
    needsSort = true;
    updateControls();
    requestFrame();
  }

  function activeShCoefficientCount() {
    if (diagnosticMode === 3) return 0;
    const bandCount = Math.min(Math.max(Math.round(renderSettings.shBandCount), 0), shCoefficientsByBand.length - 1);
    return Math.min(sourceShCoefficientCount, shCoefficientsByBand[bandCount]);
  }

  function setRenderSetting(name, value) {
    const nextSettings = { ...renderSettings };
    if (name === "toneMapping") {
      nextSettings.toneMapping = toneMappingModes.has(value) ? value : defaultRenderSettings.toneMapping;
    } else {
      nextSettings[name] = Number(value);
    }
    renderSettings = normalizedRenderSettings(nextSettings);
    if (name === "pixelScale") {
      needsResize = true;
    }
    updateControls();
    requestFrame();
  }

  function stepKeyboard(elapsed) {
    if (navigateByKeyboard(elapsed)) {
      markCameraChanged();
      requestFrame();
    }
  }

  function setPointerActive(isActive) {
    pointerIsActive = isActive;
  }

  function updateControls() {
    modeToggle.textContent = camera.settings.interactionMode === "fly" ? "Fly" : "Orbit";
    rotationToggle.setAttribute("aria-pressed", rotationIsEnabled ? "true" : "false");
    diagnosticModeSelect.value = String(diagnosticMode);
    fovControl.value = String(renderSettings.fovDegrees);
    fovValue.textContent = `${Math.round(renderSettings.fovDegrees)}°`;
    exposureControl.value = renderSettings.exposure.toFixed(2);
    exposureValue.textContent = renderSettings.exposure.toFixed(2);
    pixelScaleControl.value = renderSettings.pixelScale.toFixed(2);
    pixelScaleValue.textContent = renderSettings.pixelScale.toFixed(2);
    shBandControl.value = String(renderSettings.shBandCount);
    toneMappingControl.value = renderSettings.toneMapping;
  }

  return {
    frameScene,
    setDiagnosticMode,
    setScene,
    start,
    setRenderSetting,
    orbit,
    pan,
    zoom,
    setPointerActive,
    stepKeyboard,
    toggleMode,
    toggleRotation,
  };
}

function statusForScene(scene, pointCount, diagnosticMode = 0, framesPerSecond = 0, sortDuration = 0, sortState = "sort pending", sourceFormat = null, renderSettings = defaultRenderSettings) {
  const rendered = scene.renderedSplatCount ?? pointCount;
  const source = scene.sourceSplatCount ?? rendered;
  const diagnostic = diagnosticModeLabels.get(diagnosticMode) ?? "normal";
  const frameRate = framesPerSecond > 0 ? `${Math.round(framesPerSecond)} FPS` : "FPS pending";
  const timedSort = sortDuration > 0 ? `${sortDuration.toFixed(sortDuration >= 10 ? 0 : 1)} ms` : null;
  const sort = timedSort ? `${sortState} ${timedSort}` : sortState;
  const render = `FOV ${Math.round(renderSettings.fovDegrees)}° • exp ${renderSettings.exposure.toFixed(2)} • px ${renderSettings.pixelScale.toFixed(2)} • SH${renderSettings.shBandCount}`;
  if (scene.splatBuffer) {
    const label = sourceFormat === "splat-f32-cov-sh-palette-v1"
      ? "Swift covariance palette-SH splats"
      : "Swift covariance SH splats";
    return `WebGPU: ${rendered.toLocaleString()} / ${source.toLocaleString()} ${label} (${formatBytes(scene.splatBuffer.byteLength)}) • ${diagnostic} • ${render} • ${frameRate} • ${sort}`;
  }
  if (scene.dataSource === "swift-sog-sh0" || scene.dataSource === "swift-sog-sh0-binary" || scene.dataSource === "swift-sog-sh3-binary" || scene.dataSource === "swift-sog-sh-palette-binary") {
    return `WebGPU: ${rendered.toLocaleString()} / ${source.toLocaleString()} Swift-decoded splats • ${diagnostic} • ${render}`;
  }
  return `WebGPU: ${pointCount.toLocaleString()} Swift-generated probe splats • ${diagnostic} • ${render}`;
}

function formatBytes(byteLength) {
  if (byteLength < 1024) return `${byteLength} B`;
  const mib = byteLength / (1024 * 1024);
  return `${mib.toFixed(mib >= 10 ? 0 : 1)} MiB`;
}

function startInteraction(canvas) {
  let activePointer = null;
  let lastX = 0;
  let lastY = 0;
  let pointerMode = "orbit";

  modeToggle.addEventListener("click", () => {
    renderer?.toggleMode();
    canvas.focus({ preventScroll: true });
  });

  rotationToggle.addEventListener("click", () => {
    renderer?.toggleRotation();
    canvas.focus({ preventScroll: true });
  });

  frameButton.addEventListener("click", () => {
    renderer?.frameScene();
    canvas.focus({ preventScroll: true });
  });

  diagnosticModeSelect.addEventListener("change", () => {
    renderer?.setDiagnosticMode(Number(diagnosticModeSelect.value));
    canvas.focus({ preventScroll: true });
  });

  fovControl.addEventListener("input", () => {
    renderer?.setRenderSetting("fovDegrees", fovControl.value);
  });

  exposureControl.addEventListener("input", () => {
    renderer?.setRenderSetting("exposure", exposureControl.value);
  });

  pixelScaleControl.addEventListener("input", () => {
    renderer?.setRenderSetting("pixelScale", pixelScaleControl.value);
  });

  shBandControl.addEventListener("change", () => {
    renderer?.setRenderSetting("shBandCount", shBandControl.value);
    canvas.focus({ preventScroll: true });
  });

  toneMappingControl.addEventListener("change", () => {
    renderer?.setRenderSetting("toneMapping", toneMappingControl.value);
    canvas.focus({ preventScroll: true });
  });

  canvas.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    canvas.focus({ preventScroll: true });
    activePointer = event.pointerId;
    pointerMode = event.button === 2 || event.shiftKey ? "pan" : "orbit";
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(activePointer);
    renderer?.setPointerActive(true);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (event.pointerId !== activePointer) return;
    const deltaX = event.clientX - lastX;
    const deltaY = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    if (pointerMode === "pan") {
      renderer?.pan(deltaX, deltaY);
    } else {
      renderer?.orbit(deltaX, deltaY);
    }
  });

  canvas.addEventListener("pointerup", () => {
    activePointer = null;
    renderer?.setPointerActive(false);
  });

  canvas.addEventListener("pointercancel", () => {
    activePointer = null;
    renderer?.setPointerActive(false);
  });

  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    renderer?.zoom(event.deltaY);
  }, { passive: false });

  window.addEventListener("keydown", (event) => {
    if (!shouldHandleKeyboardEvent(event)) return;

    shiftIsPressed = event.shiftKey;
    if (event.code === "KeyR" && noCommandModifiers(event)) {
      event.preventDefault();
      if (!event.repeat) renderer?.toggleRotation();
      return;
    }
    if (event.code === "KeyF" && noCommandModifiers(event)) {
      event.preventDefault();
      if (!event.repeat) renderer?.frameScene();
      return;
    }
    if (event.code === "KeyS" && event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      if (!event.repeat) renderer?.toggleMode();
      return;
    }
    if (!navigationKeyCodes.has(event.code) || event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }

    event.preventDefault();
    pressedNavigationKeys.add(event.code);
    renderer?.stepKeyboard(1 / 60);
    renderer?.start();
  });

  window.addEventListener("keyup", (event) => {
    shiftIsPressed = event.shiftKey;
    pressedNavigationKeys.delete(event.code);
  });

  window.addEventListener("blur", () => {
    pressedNavigationKeys.clear();
    shiftIsPressed = false;
  });
}

function shouldHandleKeyboardEvent(event) {
  const target = event.target;
  if (!(target instanceof Element)) return true;
  return !target.matches("input, textarea, select, [contenteditable='true']");
}

function noCommandModifiers(event) {
  return !event.metaKey && !event.ctrlKey && !event.altKey;
}

function keyboardInput() {
  const input = {
    forward: 0,
    pitch: 0,
    speedMultiplier: shiftIsPressed ? 2 : 1,
    strafe: 0,
    vertical: 0,
    yaw: 0,
    hasMovementOrRotation: false,
  };

  if (pressedNavigationKeys.has("KeyW")) input.forward += 1;
  if (pressedNavigationKeys.has("KeyS")) input.forward -= 1;
  if (pressedNavigationKeys.has("KeyD")) input.strafe += 1;
  if (pressedNavigationKeys.has("KeyA")) input.strafe -= 1;
  if (pressedNavigationKeys.has("KeyE")) input.vertical += 1;
  if (pressedNavigationKeys.has("Space")) input.vertical += 1;
  if (pressedNavigationKeys.has("KeyQ")) input.vertical -= 1;
  if (pressedNavigationKeys.has("ArrowLeft")) input.yaw += 1;
  if (pressedNavigationKeys.has("ArrowRight")) input.yaw -= 1;
  if (pressedNavigationKeys.has("ArrowUp")) input.pitch += 1;
  if (pressedNavigationKeys.has("ArrowDown")) input.pitch -= 1;

  input.hasMovementOrRotation = input.forward !== 0
    || input.pitch !== 0
    || input.strafe !== 0
    || input.vertical !== 0
    || input.yaw !== 0;
  return input;
}

function resizeCanvas(canvas, pixelScale = 1) {
  const scale = Math.min(Math.max((window.devicePixelRatio || 1) * pixelScale, 0.25), 3);
  const width = Math.max(1, Math.floor(canvas.clientWidth * scale));
  const height = Math.max(1, Math.floor(canvas.clientHeight * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function cameraFromScene(scene) {
  const settings = {
    ...scene.cameraSettings,
    nearClipDistance: Math.pow(10, scene.cameraSettings.nearClipExponent ?? -5),
  };
  return {
    distance: scene.camera.distance,
    pitch: scene.camera.pitch,
    yaw: scene.camera.yaw,
    targetOffset: scene.camera.targetOffset,
    bounds: scene.bounds,
    settings,
  };
}

function normalizedRenderSettings(settings = {}) {
  const toneMapping = toneMappingModes.has(settings.toneMapping)
    ? settings.toneMapping
    : defaultRenderSettings.toneMapping;
  return {
    covarianceBlur: clampNumber(settings.covarianceBlur ?? defaultRenderSettings.covarianceBlur, 0, 2),
    exposure: clampNumber(settings.exposure ?? defaultRenderSettings.exposure, 0.25, 3),
    fovDegrees: clampNumber(settings.fovDegrees ?? defaultRenderSettings.fovDegrees, 25, 90),
    pixelScale: clampNumber(settings.pixelScale ?? defaultRenderSettings.pixelScale, 0.5, 1.5),
    shBandCount: Math.min(Math.max(Math.round(settings.shBandCount ?? defaultRenderSettings.shBandCount), 0), 3),
    toneMapping,
  };
}

function clampNumber(value, lowerBound, upperBound) {
  const number = Number(value);
  if (!Number.isFinite(number)) return lowerBound;
  return Math.min(Math.max(number, lowerBound), upperBound);
}

function cameraTarget(camera) {
  const radius = camera.bounds.radius;
  return {
    x: camera.bounds.center.x + camera.targetOffset.x * radius,
    y: camera.bounds.center.y + camera.targetOffset.y * radius,
    z: camera.bounds.center.z + camera.targetOffset.z * radius,
  };
}

function cameraEye(camera) {
  const target = cameraTarget(camera);
  const distance = camera.bounds.radius * camera.distance;
  return {
    x: target.x + Math.sin(camera.yaw) * Math.cos(camera.pitch) * distance,
    y: target.y + Math.sin(camera.pitch) * distance,
    z: target.z + Math.cos(camera.yaw) * Math.cos(camera.pitch) * distance,
  };
}

function cameraViewProjection(camera, aspect) {
  return multiply(cameraProjection(camera, aspect), cameraView(camera));
}

function cameraView(camera) {
  return lookAt(cameraEye(camera), cameraTarget(camera), { x: 0, y: 1, z: 0 });
}

function cameraProjection(camera, aspect, fovDegrees = 60) {
  return perspective(degreesToRadians(fovDegrees), aspect, camera.settings.nearClipDistance, 500);
}

function degreesToRadians(degrees) {
  return degrees * Math.PI / 180;
}

function perspective(fovy, aspect, near, far) {
  const yScale = 1 / Math.tan(fovy * 0.5);
  const xScale = yScale / aspect;
  const zScale = far / (near - far);
  const wzScale = zScale * near;
  return [
    xScale, 0, 0, 0,
    0, yScale, 0, 0,
    0, 0, zScale, -1,
    0, 0, wzScale, 0,
  ];
}

function lookAt(eye, center, up) {
  const z = normalize(subtract(eye, center));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [
    x.x, y.x, z.x, 0,
    x.y, y.y, z.y, 0,
    x.z, y.z, z.z, 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ];
}

function cameraBasis(camera) {
  const eye = cameraEye(camera);
  const center = cameraTarget(camera);
  const z = normalize(subtract(eye, center));
  const right = normalize(cross({ x: 0, y: 1, z: 0 }, z));
  const up = cross(z, right);
  return { right, up };
}

function multiply(lhs, rhs) {
  const result = new Array(16).fill(0);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      for (let i = 0; i < 4; i += 1) {
        result[column * 4 + row] += lhs[i * 4 + row] * rhs[column * 4 + i];
      }
    }
  }
  return result;
}

function subtract(lhs, rhs) {
  return { x: lhs.x - rhs.x, y: lhs.y - rhs.y, z: lhs.z - rhs.z };
}

function add(lhs, rhs) {
  return { x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z };
}

function scale(value, amount) {
  return { x: value.x * amount, y: value.y * amount, z: value.z * amount };
}

function length(value) {
  return Math.hypot(value.x, value.y, value.z);
}

function dot(lhs, rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

function cross(lhs, rhs) {
  return {
    x: lhs.y * rhs.z - lhs.z * rhs.y,
    y: lhs.z * rhs.x - lhs.x * rhs.z,
    z: lhs.x * rhs.y - lhs.y * rhs.x,
  };
}

function normalize(value) {
  const vectorLength = length(value);
  if (vectorLength === 0) return { x: 0, y: 0, z: 0 };
  return { x: value.x / vectorLength, y: value.y / vectorLength, z: value.z / vectorLength };
}

function cameraOrbitDirection(camera) {
  return {
    x: Math.sin(camera.yaw) * Math.cos(camera.pitch),
    y: Math.sin(camera.pitch),
    z: Math.cos(camera.yaw) * Math.cos(camera.pitch),
  };
}

function cameraForward(camera) {
  return scale(cameraOrbitDirection(camera), -1);
}

function pointerPitchDirection(settings) {
  return settings.pointerTiltYAxisIsInverted ? -1 : 1;
}
