import time
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# ---- KLT Compression Helpers ----

def compress_hsi(image_data, k):
    rows, cols, bands = image_data.shape
    pixel_vectors = image_data.reshape((rows * cols, bands))
    mean_vector = np.mean(pixel_vectors, axis=0)
    centered_data = pixel_vectors - mean_vector
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    transform_matrix = eigenvectors[:, sorted_indices[:k]]
    compressed_data = np.dot(centered_data, transform_matrix)
    return compressed_data, transform_matrix, mean_vector

def reconstruct_image(compressed_data, transform_matrix, mean_vector, original_shape):
    reconstructed_centered = np.dot(compressed_data, transform_matrix.T)
    reconstructed_pixels = reconstructed_centered + mean_vector
    return reconstructed_pixels.reshape(original_shape)

def calculate_metrics(original_data, reconstructed_data, k):
    _, _, original_bands = original_data.shape
    compression_ratio = original_bands / k
    mse = np.mean((original_data - reconstructed_data) ** 2)
    psnr = float("inf") if mse == 0 else 20 * np.log10(np.max(original_data) / np.sqrt(mse))
    return {
        "compression_ratio": f"{compression_ratio:.2f}:1",
        "psnr_db": round(psnr, 2),
    }

# ---- API Routes ----

@app.route("/api/health")
def health():
    return jsonify({"message": "KLT Backend API is running on Vercel."})

@app.route("/api/compress", methods=["POST"])
def compress_endpoint():
    start_time = time.time()
    image_file = request.files.get("image_file")
    k_components_str = request.form.get("k_components")

    if not image_file:
        return jsonify({"error": "No image_file uploaded"}), 400

    # ---- Image compression (works on Vercel) ----
    if image_file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            img = Image.open(image_file.stream)
            buf = io.BytesIO()
            if image_file.filename.lower().endswith(".png"):
                img.save(buf, format="PNG", optimize=True)
            else:
                img.save(buf, format="JPEG", quality=75, optimize=True)

            compressed_bytes = buf.getvalue()
            metrics = {
                "original_size_bytes": image_file.content_length,
                "compressed_size_bytes": len(compressed_bytes),
                "compression_ratio": f"{image_file.content_length/len(compressed_bytes):.2f}:1"
            }
            metrics["processing_time_sec"] = round(time.time() - start_time, 2)

            return jsonify({
                "metrics": metrics,
                "compressed_image": compressed_bytes.hex()
            })
        except Exception as e:
            return jsonify({"error": f"Failed to compress image: {str(e)}"}), 500

    # ---- Hyperspectral files (not supported on Vercel) ----
    if image_file.filename.lower().endswith((".hdr", ".raw")):
        return jsonify({
            "error": "Hyperspectral (.hdr/.raw) processing requires filesystem access. "
                     "Use external storage (S3, Supabase, etc.) or deploy to Railway/Render."
        }), 501

    return jsonify({"error": "Unsupported file type"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
