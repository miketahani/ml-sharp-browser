import * as ort from "onnxruntime-web/webgpu";

const INPUT_SIZE = 1536;
const MODEL_PATH = "/sharp.onnx";

export interface GaussianOutput {
  meanVectors: ort.Tensor;
  singularValues: ort.Tensor;
  quaternions: ort.Tensor;
  colors: ort.Tensor;
  opacities: ort.Tensor;
  numGaussians: number;
}

let session: ort.InferenceSession | null = null;

export async function loadModel(
  onProgress?: (msg: string) => void
): Promise<void> {
  if (session) return;

  onProgress?.("Initializing WebGPU...");
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/";
  ort.env.wasm.numThreads = 1; // Avoid SharedArrayBuffer requirement (GitHub Pages can't set COOP/COEP headers)

  onProgress?.("Loading model...");
  session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ["webgpu", "wasm"],
    externalData: [
      {
        path: "sharp.onnx.data",
        data: MODEL_PATH + ".data",
      },
    ],
  });
  onProgress?.("Model loaded.");
}

/**
 * Preprocess an image: resize to 1536x1536, normalize to [0,1], CHW format.
 */
function preprocessImage(img: HTMLImageElement): Float32Array {
  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = imageData;

  // Convert RGBA HWC uint8 -> RGB CHW float32 [0, 1]
  const chw = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const pixelCount = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < pixelCount; i++) {
    chw[i] = data[i * 4] / 255; // R
    chw[pixelCount + i] = data[i * 4 + 1] / 255; // G
    chw[2 * pixelCount + i] = data[i * 4 + 2] / 255; // B
  }
  return chw;
}

export async function runInference(
  img: HTMLImageElement,
  focalLengthPx?: number
): Promise<GaussianOutput> {
  if (!session) throw new Error("Model not loaded");

  const width = img.naturalWidth;
  const height = img.naturalHeight;
  // Default focal length estimate: assume ~60° horizontal FOV
  const fPx = focalLengthPx ?? width * 0.85;
  const disparityFactor = fPx / width;

  const imageData = preprocessImage(img);
  const imageTensor = new ort.Tensor("float32", imageData, [
    1,
    3,
    INPUT_SIZE,
    INPUT_SIZE,
  ]);
  const disparityTensor = new ort.Tensor(
    "float32",
    Float32Array.of(disparityFactor),
    [1]
  );

  const results = await session.run({
    image: imageTensor,
    disparity_factor: disparityTensor,
  });

  return {
    meanVectors: results.mean_vectors,
    singularValues: results.singular_values,
    quaternions: results.quaternions,
    colors: results.colors,
    opacities: results.opacities,
    numGaussians: results.mean_vectors.dims[1],
  };
}
