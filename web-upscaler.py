# Standard library imports
import os
import io
import gc
import re
import time
import asyncio
import traceback
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import threading
import concurrent.futures
from functools import partial
import json
import configparser
import logging

# Third-party library imports
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
import spandrel
import spandrel_extra_arches
from flask import Flask, request, jsonify, render_template, send_file, Response
from werkzeug.utils import secure_filename

# Local module imports
from utils.vram_estimator import estimate_vram_and_tile_size, get_free_vram
from utils.fuzzy_model_matcher import find_closest_models, search_models
from utils.alpha_handler import handle_alpha
from utils.resize_module import resize_image, get_available_filters, MIN_SCALE_FACTOR, MAX_SCALE_FACTOR
from utils.image_info import get_image_info

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Install extra architectures
spandrel_extra_arches.install()

app = Flask(__name__)

# Configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Global variables
models = {}
thread_pool = ThreadPoolExecutor(max_workers=int(config['Processing'].get('ThreadPoolWorkers', 1)))
last_cleanup_time = time.time()
CLEANUP_INTERVAL = 3 * 60 * 60  # 3 hours in seconds

# Configuration management
def load_config():
    config.read('config.ini')
    return {section: dict(config[section]) for section in config.sections()}

def save_config(new_config):
    for section in new_config:
        if section not in config:
            config.add_section(section)
        for key, value in new_config[section].items():
            config[section][key] = str(value)
    
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

# Model management
def load_model(model_name):
    if model_name in models:
        return models[model_name]
    
    model_path = os.path.join(config['Paths']['ModelPath'], f"{model_name}")
    if os.path.exists(model_path + '.pth'):
        model_path = model_path + '.pth'
    elif os.path.exists(model_path + '.safetensors'):
        model_path = model_path + '.safetensors'
    else:
        raise ValueError(f"Model file not found: {model_name}")
    
    try:
        model = spandrel.ModelLoader().load_from_file(model_path)
        if isinstance(model, spandrel.ImageModelDescriptor):
            models[model_name] = model.cuda().eval()
            return models[model_name]
        else:
            raise ValueError(f"Invalid model type for {model_name}")
    except Exception as e:
        raise

def list_available_models():
    model_path = config['Paths']['ModelPath']
    return [os.path.splitext(f)[0] for f in os.listdir(model_path) 
            if f.endswith(('.pth', '.safetensors'))]

# Image processing functions
def upscale_image(image, model, tile_size, alpha_handling, has_alpha, precision, check_cancelled):
    try:
        logger.debug(f"Starting upscale with tile size: {tile_size}, precision: {precision}")
        
        def upscale_func(img):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).cuda()
            _, _, h, w = img_tensor.shape
            output_h, output_w = h * model.scale, w * model.scale
            logger.debug(f"Processing image: {w}x{h} -> {output_w}x{output_h}")

            if model.supports_bfloat16 and precision in ['auto', 'bf16']:
                output_dtype = torch.bfloat16
                autocast_dtype = torch.bfloat16
            elif model.supports_half and precision in ['auto', 'fp16']:
                output_dtype = torch.float16
                autocast_dtype = torch.float16
            else:
                output_dtype = torch.float32
                autocast_dtype = None

            logger.debug(f"Using precision mode: {autocast_dtype}")

            output_tensor = torch.zeros((1, img_tensor.shape[1], output_h, output_w), 
                                     dtype=output_dtype, device='cuda')

            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    if check_cancelled():
                        raise asyncio.CancelledError("Upscale operation was cancelled")

                    tile = img_tensor[:, :, y:min(y+tile_size, h), x:min(x+tile_size, w)]
                    with torch.inference_mode():
                        if autocast_dtype:
                            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                                upscaled_tile = model(tile)
                        else:
                            upscaled_tile = model(tile)
                    
                    output_tensor[:, :, 
                                y*model.scale:min((y+tile_size)*model.scale, output_h),
                                x*model.scale:min((x+tile_size)*model.scale, output_w)].copy_(upscaled_tile)

            return Image.fromarray((output_tensor[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8))

        if has_alpha:
            return handle_alpha(image, upscale_func, alpha_handling, 
                              config['Processing'].getboolean('GammaCorrection', False))
        else:
            return upscale_func(image)

    except Exception as e:
        logger.error(f"Error in upscale_image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_image_metadata(image):
    """Get metadata for an image object."""
    info = {
        'format': image.format,
        'mode': image.mode,
        'size': {'width': image.width, 'height': image.height},
        'pixels': image.width * image.height,
        'megapixels': (image.width * image.height) / 1_000_000
    }
    
    # Add additional metadata if available
    if hasattr(image, 'info'):
        if 'dpi' in image.info:
            info['dpi'] = image.info['dpi']
        if 'icc_profile' in image.info:
            info['has_icc_profile'] = True
        if 'exif' in image.info:
            info['has_exif'] = True
    
    # Check for alpha channel
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        info['has_alpha'] = True
    else:
        info['has_alpha'] = False
    
    # Estimate memory usage
    estimated_memory = (image.width * image.height * len(image.getbands()) * 8) / (8 * 1024 * 1024)  # in MB
    info['estimated_memory'] = f"{estimated_memory:.2f} MB"
    
    return info

# Routes
@app.route('/')
def index():
    return render_template('index.html', 
                         models=list_available_models(),
                         config=load_config())

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'GET':
        return jsonify(load_config())
    else:
        new_config = request.json
        save_config(new_config)
        return jsonify({"status": "success"})

@app.route('/api/models')
def get_models():
    return jsonify(list_available_models())

@app.route('/api/upscale', methods=['POST'])
def upscale():
    try:
        logger.debug("Starting upscale request")
        
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        model_name = request.form.get('model')
        alpha_handling = request.form.get('alpha_handling', 
                                        config['Processing']['DefaultAlphaHandling'])
        
        if not model_name:
            return jsonify({"error": "No model specified"}), 400
        
        logger.debug(f"Processing upscale request: model={model_name}, alpha={alpha_handling}")
        
        # Load and process image
        image = Image.open(file.stream)
        
        # Convert RGBA images to RGB if alpha handling is 'discard'
        if alpha_handling == 'discard' and image.mode == 'RGBA':
            image = image.convert('RGB')
        
        has_alpha = image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info)
        
        # Load model
        try:
            model = load_model(model_name)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
        
        # Calculate tile size and VRAM
        try:
            input_size = (image.width, image.height)
            estimated_vram, adjusted_tile_size = estimate_vram_and_tile_size(
                model=model,
                input_size=input_size
            )
            
            logger.debug(f"VRAM estimation complete. Tile size: {adjusted_tile_size}")
            
        except Exception as e:
            logger.error(f"Error estimating VRAM: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Failed to estimate VRAM requirements: {str(e)}"}), 500
        
        # Process image
        try:
            result = upscale_image(
                image=image,
                model=model,
                tile_size=adjusted_tile_size,
                alpha_handling=alpha_handling,
                has_alpha=has_alpha,
                precision=config['Processing'].get('Precision', 'auto').lower(),
                check_cancelled=lambda: False
            )
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory error")
            return jsonify({"error": "Not enough GPU memory to process the image. Try a smaller image or different model."}), 500
        except Exception as e:
            logger.error(f"Error during upscaling: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Failed to upscale image: {str(e)}"}), 500
        
        # Save result
        try:
            output = io.BytesIO()
            save_format = 'PNG'
            result.save(output, format=save_format)
            output.seek(0)
            
            return send_file(
                output,
                mimetype='image/png',
                as_attachment=True,
                download_name=f"upscaled_{file.filename}"
            )
        except Exception as e:
            logger.error(f"Error saving result: {str(e)}")
            return jsonify({"error": f"Failed to save result: {str(e)}"}), 500
        
    except Exception as e:
        logger.error(f"Error in upscale endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

@app.route('/api/resize', methods=['POST'])
def resize():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        scale_factor = float(request.form.get('scale_factor', 1.0))
        method = request.form.get('method', 'box')
        gamma_correction = config['Processing'].getboolean('GammaCorrection', False)
        
        # Validate scale factor
        if scale_factor < MIN_SCALE_FACTOR or scale_factor > MAX_SCALE_FACTOR:
            return jsonify({
                "error": f"Scale factor must be between {MIN_SCALE_FACTOR} and {MAX_SCALE_FACTOR} (inclusive)."
            }), 400
        
        # Validate resize method
        available_filters = get_available_filters()
        if method.lower() not in available_filters:
            return jsonify({
                "error": f"Unsupported method: {method}",
                "available_methods": available_filters
            }), 400
        
        image = Image.open(file.stream)
        logger.debug(f"Original image size: {image.size}")
        logger.debug(f"Original image mode: {image.mode}")
        logger.debug(f"Scale factor: {scale_factor}")
        logger.debug(f"Method: {method}")
        
        # Perform resizing using the full resize_image function from resize_module
        result = resize_image(
            image=image,
            scale_factor=scale_factor,
            method=method,
            gamma_correction=gamma_correction
        )
        
        logger.debug(f"Resized image size: {result.size}")
        logger.debug(f"Resized image mode: {result.mode}")
        
        # Save result
        output = io.BytesIO()
        result.save(output, format='PNG')
        output.seek(0)
        
        # Create descriptive filename
        operation = "upscaled" if scale_factor > 1 else "downscaled"
        new_filename = f"{operation}_{scale_factor}x_{method}_{file.filename}"
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=new_filename
        )
        
    except ValueError as e:
        logger.error(f"Value error in resize endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in resize endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/resize/methods', methods=['GET'])
def get_resize_methods():
    """Get available resize methods and scale factor limits."""
    return jsonify({
        'methods': get_available_filters(),
        'scale_limits': {
            'min': MIN_SCALE_FACTOR,
            'max': MAX_SCALE_FACTOR
        }
    })

@app.route('/api/info', methods=['POST'])
def get_info():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        image = Image.open(file.stream)
        
        # Get image information
        info = get_image_metadata(image)
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error in info endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Cleanup task
def cleanup_models_task():
    global models, last_cleanup_time
    while True:
        time.sleep(60)  # Check every minute
        current_time = time.time()
        if current_time - last_cleanup_time >= CLEANUP_INTERVAL:
            print("Performing periodic cleanup of unused models...")
            models.clear()
            torch.cuda.empty_cache()
            gc.collect()
            last_cleanup_time = current_time
            print("Cache cleanup completed. All models unloaded and memory freed.")

# Default configuration
DEFAULT_CONFIG = {
    'Paths': {
        'ModelPath': 'models'
    },
    'Processing': {
        'MaxTileSize': '1024',
        'Precision': 'auto',
        'MaxOutputTotalPixels': '67108864',
        'UpscaleTimeout': '60',
        'OtherStepTimeout': '30',
        'ThreadPoolWorkers': '1',
        'MaxConcurrentUpscales': '1',
        'DefaultAlphaHandling': 'resize',
        'GammaCorrection': 'false',
        'VRAMSafetyMultiplier': '1.2',
        'AvailableVRAMUsageFraction': '0.8',
        'DefaultTileSize': '384'
    }
}

# Initialize config with defaults if it doesn't exist
def init_config():
    if not os.path.exists('config.ini'):
        for section, values in DEFAULT_CONFIG.items():
            if section not in config:
                config.add_section(section)
            for key, value in values.items():
                config[section][key] = value
        
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

if __name__ == '__main__':
    cleanup_thread = threading.Thread(target=cleanup_models_task, daemon=True)
    cleanup_thread.start()
    app.run(host='localhost', port=5000, debug=True)
