import { useState, useRef, useCallback } from "react";
import { loadModel, runInference } from "./inference";
import { unprojectGaussians } from "./postprocess";
import { createViewer } from "./viewer";
import "./App.css";

type Status = "idle" | "loading-model" | "running" | "postprocessing" | "done" | "error";

function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [stats, setStats] = useState<string | null>(null);
  const viewerRef = useRef<HTMLDivElement>(null);
  const viewerDispose = useRef<(() => void) | null>(null);
  const imageUrlRef = useRef<string | null>(null);

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
        setMessage("Loading model...");
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
          `${gaussians.count.toLocaleString()} Gaussians | inference ${(inferenceTime / 1000).toFixed(1)}s`
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
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", fontFamily: "system-ui" }}>
      <div style={{ padding: "12px 20px", borderBottom: "1px solid #333", background: "#1a1a1a", display: "flex", alignItems: "center", gap: 16 }}>
        <h1 style={{ margin: 0, fontSize: 16, color: "#eee" }}>ML-SHARP Browser</h1>

        <label
          htmlFor="image-input"
          style={{
            padding: "6px 14px",
            background: "#2563eb",
            color: "white",
            borderRadius: 6,
            cursor: status === "loading-model" || status === "running" ? "not-allowed" : "pointer",
            fontSize: 13,
            opacity: status === "loading-model" || status === "running" ? 0.5 : 1,
          }}
        >
          {status === "idle" ? "Open image" : "New image"}
        </label>
        <input
          id="image-input"
          type="file"
          accept="image/*"
          onChange={handleFile}
          style={{ display: "none" }}
          disabled={status === "loading-model" || status === "running"}
        />

        {message && (
          <span style={{ fontSize: 13, color: status === "error" ? "#f87171" : "#93c5fd" }}>
            {message}
          </span>
        )}

        {stats && (
          <span style={{ fontSize: 13, color: "#9ca3af", marginLeft: "auto" }}>
            {stats}
          </span>
        )}
      </div>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Input image thumbnail */}
        {imageUrl && (
          <div style={{ width: 240, borderRight: "1px solid #333", background: "#111", padding: 12, overflow: "auto" }}>
            <img
              src={imageUrl}
              alt="Input"
              style={{ width: "100%", borderRadius: 6 }}
            />
          </div>
        )}

        {/* 3D viewer */}
        <div
          ref={viewerRef}
          style={{
            flex: 1,
            background: "#111",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#555",
            fontSize: 14,
          }}
        >
          {status === "idle" && "Select an image to begin"}
        </div>
      </div>
    </div>
  );
}

export default App;
