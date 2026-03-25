import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SplatMesh } from "@sparkjsdev/spark";
import type { Gaussians3D } from "./postprocess";

export function createViewer(
  container: HTMLElement,
  gaussians: Gaussians3D
): { dispose: () => void } {
  const width = container.clientWidth;
  const height = container.clientHeight;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 100);
  camera.position.set(0, 0, 3);
  camera.lookAt(0, 0, 0);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;

  const { means, scales, quaternions, colors, opacities, count } = gaussians;

  const splatMesh = new SplatMesh({
    constructSplats: (splats) => {
      const center = new THREE.Vector3();
      const scale = new THREE.Vector3();
      const quat = new THREE.Quaternion();
      const color = new THREE.Color();

      for (let i = 0; i < count; i++) {
        center.set(means[i * 3], means[i * 3 + 1], means[i * 3 + 2]);
        scale.set(scales[i * 3], scales[i * 3 + 1], scales[i * 3 + 2]);
        // THREE.Quaternion uses (x, y, z, w), SHARP uses (w, x, y, z)
        quat.set(
          quaternions[i * 4 + 1],
          quaternions[i * 4 + 2],
          quaternions[i * 4 + 3],
          quaternions[i * 4]
        );
        color.setRGB(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]);

        splats.pushSplat(center, scale, quat, opacities[i], color);
      }
    },
  });

  scene.add(splatMesh);

  // Compute median center, then use 90th percentile distance to set bounds
  // (ignores far-flung outlier splats)
  const sampleStep = Math.max(1, Math.floor(count / 10000));
  const sampleCount = Math.ceil(count / sampleStep);
  const xs = new Float32Array(sampleCount);
  const ys = new Float32Array(sampleCount);
  const zs = new Float32Array(sampleCount);
  for (let i = 0, j = 0; i < count; i += sampleStep, j++) {
    xs[j] = means[i * 3];
    ys[j] = means[i * 3 + 1];
    zs[j] = means[i * 3 + 2];
  }
  xs.sort();
  ys.sort();
  zs.sort();
  const mid = Math.floor(sampleCount / 2);
  const median = new THREE.Vector3(xs[mid], ys[mid], zs[mid]);

  // Compute distances from median, use 90th percentile as radius
  const dists = new Float32Array(sampleCount);
  for (let i = 0, j = 0; i < count; i += sampleStep, j++) {
    const dx = means[i * 3] - median.x;
    const dy = means[i * 3 + 1] - median.y;
    const dz = means[i * 3 + 2] - median.z;
    dists[j] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  dists.sort();
  const radius = dists[Math.floor(sampleCount * 0.9)];

  camera.position.copy(median).add(new THREE.Vector3(0, 0, radius * 1.2));
  controls.target.copy(median);
  controls.update();

  let animationId: number;
  function animate() {
    animationId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  const handleResize = () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  };
  window.addEventListener("resize", handleResize);

  return {
    dispose: () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", handleResize);
      controls.dispose();
      splatMesh.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    },
  };
}
