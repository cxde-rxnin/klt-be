import os
import io
import time
import logging
import numpy as np
from PIL import Image
from werkzeug.datastructures import FileStorage
from werkzeug.wrappers import Request, Response
import json

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.INFO)

def handler(request: Request):
    start_time = time.time()
    # Parse multipart form data
    form = request.form
    files = request.files
    image_file: FileStorage = files.get('image_file')
    profile = form.get('profile')
    k_components_str = form.get('k_components')
    k_components = int(k_components_str) if k_components_str else None
    if image_file and image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            img = Image.open(image_file.stream)
            buf = io.BytesIO()
            if image_file.filename.lower().endswith('.png'):
                img.save(buf, format='PNG', optimize=True)
            else:
                img.save(buf, format='JPEG', quality=75, optimize=True)
            compressed_bytes = buf.getvalue()
            image_file.stream.seek(0, os.SEEK_END)
            original_size = image_file.stream.tell()
            compressed_size = len(compressed_bytes)
            metrics = {
                "original_size_bytes": original_size,
                "compressed_size_bytes": compressed_size,
                "compression_ratio": f"{original_size/compressed_size:.2f}:1",
                "processing_time_sec": round(time.time() - start_time, 2)
            }
            response_data = {
                "metrics": metrics,
                "compressed_image": compressed_bytes.hex()
            }
            return Response(json.dumps(response_data), mimetype="application/json")
        except Exception as e:
            return Response(json.dumps({"error": f"Failed to compress image: {str(e)}"}), status=500, mimetype="application/json")
    return Response(json.dumps({"error": "Only JPG/PNG supported in serverless version."}), status=400, mimetype="application/json")
