import { useState, useRef, useCallback } from "react";
import { loadModel, runInference } from "./inference";
import { unprojectGaussians } from "./postprocess";
import { createViewer } from "./viewer";
import "./App.css";

type Status =
  | "idle"
  | "loading-model"
  | "running"
  | "postprocessing"
  | "done"
  | "error";

function InfoModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          &times;
        </button>

        <h1>ML-Sharp Browser</h1>
        <p>Requires Chromium-based browser (<a href="https://caniuse.com/?search=webgpu" target="_blank" rel="noreferrer">for now</a>)</p>
        <p>
          Browser-based inference of{" "}
          <a href="https://github.com/apple/ml-sharp" target="_blank" rel="noreferrer">Apple's SHARP</a>{" "}
          single-image 3D Gaussian prediction model, running via ONNX. Heavily vibe-coded proof of concept.
        </p>
        <p>
          The model was exported to ONNX then uploaded to HuggingFace (<a href="https://huggingface.co/mxtx0123/ml-sharp-onnx" target="_blank" rel="noreferrer">link</a>). It's approximately 2.6gb.
        </p>
        <p>
          A React app is included which allows a user to convert their own images. Select a photo and the app runs the full SHARP prediction pipeline in your browser. Image is processed by a ViT encoder, decoded into 3D splats, and rendered as an interactive splat scene. No images are uploaded anywhere and inference happens 100% clientside.
        </p>

        <p>
          <a href="https://github.com/miketahani/ml-sharp-browser" target="_blank" rel="noreferrer">
            Source on GitHub
          </a>
        </p>
      </div>
    </div>
  );
}

function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [stats, setStats] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(false);
  const viewerRef = useRef<HTMLDivElement>(null);
  const viewerDispose = useRef<(() => void) | null>(null);
  const imageUrlRef = useRef<string | null>(null);

  const isProcessing = status === "loading-model" || status === "running";

  const handleFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      // Revoke previous blob URL
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);

      const url = URL.createObjectURL(file);
      imageUrlRef.current = url;
      setImageUrl(url);
      setStats(null);

      // Clean up previous viewer
      if (viewerDispose.current) {
        viewerDispose.current();
        viewerDispose.current = null;
      }

      // Wait for image to load
      const img = new Image();
      img.src = url;
      await new Promise((resolve) => (img.onload = resolve));

      try {
        setStatus("loading-model");
        setMessage("Loading model (approx 2.6GB)...");
        await loadModel((msg) => setMessage(msg));

        setStatus("running");
        setMessage("Running inference...");
        const t0 = performance.now();
        const output = await runInference(img);
        const inferenceTime = performance.now() - t0;

        setStatus("postprocessing");
        setMessage("Unprojecting to 3D...");

        // Estimate focal length (~60° horizontal FOV)
        const fPx = img.naturalWidth * 0.85;

        const gaussians = unprojectGaussians(
          output.meanVectors.data as Float32Array,
          output.singularValues.data as Float32Array,
          output.quaternions.data as Float32Array,
          output.colors.data as Float32Array,
          output.opacities.data as Float32Array,
          output.numGaussians,
          img.naturalWidth,
          img.naturalHeight,
          fPx
        );

        // Release ONNX tensors (may hold WebGPU buffers)
        output.meanVectors.dispose();
        output.singularValues.dispose();
        output.quaternions.dispose();
        output.colors.dispose();
        output.opacities.dispose();

        setMessage("Rendering...");
        if (viewerRef.current) {
          const viewer = createViewer(viewerRef.current, gaussians);
          viewerDispose.current = viewer.dispose;
        }

        setStatus("done");
        setStats(
          `${gaussians.count.toLocaleString()} Gaussians | inference ${(
            inferenceTime / 1000
          ).toFixed(1)}s`
        );
        setMessage("");
      } catch (err) {
        setStatus("error");
        setMessage(`Error: ${err instanceof Error ? err.message : err}`);
        console.error(err);
      }
    },
    []
  );

  return (
    <div className="app">
      <div className="toolbar">
        <h1>ML-SHARP Browser</h1>

        <label
          htmlFor="image-input"
          className="file-label"
          style={{
            cursor: isProcessing ? "not-allowed" : "pointer",
            opacity: isProcessing ? 0.5 : 1,
          }}
        >
          {status === "idle" ? "Open image" : "New image"}
        </label>
        <input
          id="image-input"
          type="file"
          accept="image/*"
          onChange={handleFile}
          className="file-input"
          disabled={isProcessing}
        />

        {message && (
          <span
            className="message"
            style={{ color: status === "error" ? "#f87171" : "#93c5fd" }}
          >
            {message}
          </span>
        )}

        {stats && <span className="stats">{stats}</span>}

        <button className="info-button" onClick={() => setShowInfo(true)} title="Info">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="16" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>
        </button>
      </div>

      {showInfo && <InfoModal onClose={() => setShowInfo(false)} />}

      <div className="main">
        {imageUrl && (
          <img src={imageUrl} alt="Input" className="input-preview" />
        )}
        <div ref={viewerRef} className="viewer">
          {status === "idle" && "Select an image to begin"}
        </div>
      </div>
    </div>
  );
}

export default App;
