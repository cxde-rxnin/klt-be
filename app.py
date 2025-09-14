import os
import time
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from spectral import open_image
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024 
logging.basicConfig(level=logging.INFO)

def read_hsi_data(hdr_path):
    try:
        logging.info(f"Reading HSI data from: {hdr_path}")
        img = open_image(hdr_path)
        data = img.load()
        metadata = img.metadata
        logging.info(f"Successfully loaded image with shape: {data.shape}")
        return data, metadata
    except Exception as e:
        logging.error(f"Error reading HSI data with spectral library: {e}")
        return None, None

def compress_hsi(image_data, k):
    rows, cols, bands = image_data.shape
    pixel_vectors = image_data.reshape((rows * cols, bands))
    mean_vector = np.mean(pixel_vectors, axis=0)
    centered_data = pixel_vectors - mean_vector
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    transform_matrix = sorted_eigenvectors[:, :k]
    compressed_data = np.dot(centered_data, transform_matrix)
    logging.info(f"KLT compression complete. Projected data shape: {compressed_data.shape}")
    return compressed_data, transform_matrix, mean_vector

def reconstruct_image(compressed_data, transform_matrix, mean_vector, original_shape):
    reconstructed_centered = np.dot(compressed_data, transform_matrix.T)
    reconstructed_pixels = reconstructed_centered + mean_vector
    return reconstructed_pixels.reshape(original_shape)

def calculate_metrics(original_data, reconstructed_data, k):
    _, _, original_bands = original_data.shape
    compression_ratio = original_bands / k
    mse = np.mean((original_data - reconstructed_data) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel_val = np.max(original_data)
        psnr = 20 * np.log10(max_pixel_val / np.sqrt(mse))
    rows, cols, bands = original_data.shape
    original_pixels = original_data.reshape((rows * cols, bands))
    reconstructed_pixels = reconstructed_data.reshape((rows * cols, bands))
    dot_product = np.sum(original_pixels * reconstructed_pixels, axis=1)
    norm_orig = np.linalg.norm(original_pixels, axis=1)
    norm_recon = np.linalg.norm(reconstructed_pixels, axis=1)
    cos_angle = np.zeros_like(dot_product, dtype=float)
    valid_indices = (norm_orig > 0) & (norm_recon > 0)
    cos_angle[valid_indices] = dot_product[valid_indices] / (norm_orig[valid_indices] * norm_recon[valid_indices])
    cos_angle = np.clip(cos_angle, -1, 1)
    spectral_angles = np.arccos(cos_angle) * (180.0 / np.pi)
    mean_sam = np.mean(spectral_angles)
    return {
        "compression_ratio": f"{compression_ratio:.2f}:1",
        "psnr_db": round(psnr, 2),
        "mean_sam_degrees": round(mean_sam, 4)
    }

@app.route('/api/compress', methods=['POST'])
def compress_endpoint():
    start_time = time.time()
    hdr_file = request.files.get('hdr_file')
    raw_file = request.files.get('raw_file')
    image_file = request.files.get('image_file')
    k_components_str = request.form.get('k_components')
    profile = request.form.get('profile')
    k_components = None
    if k_components_str:
        try:
            k_components = int(k_components_str)
        except Exception:
            k_components = None
    if image_file and image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            img = Image.open(image_file)
            buf = io.BytesIO()
            if image_file.filename.lower().endswith('.png'):
                img.save(buf, format='PNG', optimize=True)
            else:
                img.save(buf, format='JPEG', quality=75, optimize=True)
            compressed_bytes = buf.getvalue()
            original_size = image_file.content_length
            compressed_size = len(compressed_bytes)
            metrics = {
                "original_size_bytes": original_size,
                "compressed_size_bytes": compressed_size,
                "compression_ratio": f"{original_size/compressed_size:.2f}:1"
            }
            processing_time = time.time() - start_time
            metrics["processing_time_sec"] = round(processing_time, 2)
            response_data = {
                "metrics": metrics,
                "compressed_image": compressed_bytes.hex()
            }
            return jsonify(response_data)
        except Exception as e:
            return jsonify({"error": f"Failed to compress image: {str(e)}"}), 500
    hdr_path = raw_path = None
    if hdr_file:
        hdr_filename = secure_filename(hdr_file.filename)
        hdr_path = os.path.join(app.config['UPLOAD_FOLDER'], hdr_filename)
        hdr_file.save(hdr_path)
    if raw_file:
        raw_filename = secure_filename(raw_file.filename)
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
        raw_file.save(raw_path)
    try:
        if hdr_path:
            original_data, metadata = read_hsi_data(hdr_path)
            if original_data is None:
                return jsonify({"error": "Failed to read hyperspectral image data. Check file format and paths."}), 500
        elif raw_path:
            presets = {
                "default": {"shape": (512, 512, 10), "dtype": "uint16"},
                "cameraA": {"shape": (1024, 1024, 16), "dtype": "uint16"},
                "cameraB": {"shape": (2048, 2048, 32), "dtype": "float32"},
            }
            if profile and profile in presets:
                shape = presets[profile]["shape"]
                dtype = presets[profile]["dtype"]
            else:
                if raw_filename.lower().endswith('.nef'):
                    shape = (512, 512, 3)
                    dtype = "uint16"
                else:
                    shape = presets["default"]["shape"]
                    dtype = presets["default"]["dtype"]
            try:
                import numpy as np
                with open(raw_path, 'rb') as f:
                    raw = np.frombuffer(f.read(), dtype=dtype)
                expected_size = shape[0] * shape[1] * shape[2]
                if raw.size != expected_size:
                    return jsonify({"error": f"RAW file size does not match expected shape {shape}."}), 400
                original_data = raw.reshape(shape)
                metadata = {"source": "RAW-only", "shape": shape, "dtype": dtype}
            except Exception as e:
                return jsonify({"error": f"Failed to read RAW file: {str(e)}"}), 500
        else:
            return jsonify({"error": "No valid file for processing."}), 400
        if k_components > original_data.shape[2]:
            return jsonify({"error": f"k_components ({k_components}) cannot be greater than the number of bands ({original_data.shape[2]})."}), 400
        compressed_data, transform_matrix, mean_vector = compress_hsi(original_data, k_components)
        reconstructed_data = reconstruct_image(compressed_data, transform_matrix, mean_vector, original_data.shape)
        metrics = calculate_metrics(original_data, reconstructed_data, k_components)
    except Exception as e:
        logging.error(f"An error occurred during KLT processing: {e}")
        return jsonify({"error": "An internal error occurred during compression."}), 500
    finally:
        if hdr_path and os.path.exists(hdr_path): os.remove(hdr_path)
        if raw_path and os.path.exists(raw_path): os.remove(raw_path)
        logging.info("Cleaned up temporary files.")
    processing_time = time.time() - start_time
    metrics["processing_time_sec"] = round(processing_time, 2)
    default_band_index = int(metadata.get('default bands', [0])[0]) if metadata and 'default bands' in metadata else 0
    if default_band_index >= reconstructed_data.shape[2]:
        default_band_index = 0
    response_data = {
        "metrics": metrics,
        "metadata": {
            "original_shape": original_data.shape,
            "data_type": str(original_data.dtype)
        },
        "reconstructed_image_preview": reconstructed_data[:, :, default_band_index].tolist(),
    }
    logging.info("Successfully processed request. Sending response.")
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)