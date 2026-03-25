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

function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [stats, setStats] = useState<string | null>(null);
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
      </div>

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
