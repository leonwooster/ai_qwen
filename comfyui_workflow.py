import json
import urllib.request
import urllib.parse
import random
import os
from PIL import Image
import io
import base64

# ComfyUI server configuration
COMFYUI_SERVER = "127.0.0.1:8188"
CLIENT_ID = str(random.randint(1, 1000000))


def queue_prompt(prompt, client_id):
    """Send a prompt to ComfyUI server"""
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{COMFYUI_SERVER}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    """Retrieve generated image from ComfyUI"""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{COMFYUI_SERVER}/view?{url_values}") as response:
        return response.read()


def upload_image(image_path):
    """Upload image to ComfyUI server"""
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/png')}
        # Note: This is a simplified version. You may need to use requests library for proper multipart upload
        # For now, we'll assume the image is already in ComfyUI's input folder
        return os.path.basename(image_path)


# Workflow definition based on the ComfyUI screenshot
workflow = {
    "1": {
        "inputs": {
            "unet_name": "Qwen-Image-Edit-2509-flux_unet.safetensors",
            "weight_dtype": "default"
        },
        "class_type": "UnetLoader (GGUF)",
        "_meta": {"title": "UnetLoader (GGUF)"}
    },
    "2": {
        "inputs": {
            "clip_name": "Qwen-Image-Lightning-8step-v1-qwen2vl_clip.safetensors",
            "type": "flux"
        },
        "class_type": "CLIPLoader (GGUF)",
        "_meta": {"title": "CLIPLoader (GGUF)"}
    },
    "3": {
        "inputs": {
            "vae_name": "qwen_image_edit_vae.safetensors"
        },
        "class_type": "Load VAE",
        "_meta": {"title": "Load VAE"}
    },
    "4": {
        "inputs": {
            "image": "input.png",  # Your input image
            "upload": "image"
        },
        "class_type": "Load Image",
        "_meta": {"title": "Load Image"}
    },
    "5": {
        "inputs": {
            "text": "make green eyes WHEN GREEN ONLY on the head to face",
            "clip": ["2", 0]
        },
        "class_type": "TextEncodeCLIPSimple(M)",
        "_meta": {"title": "TextEncodeCLIPSimple(M)"}
    },
    "6": {
        "inputs": {
            "model": ["1", 0],
            "max_shift": 1.15,
            "base_shift": 0.5,
            "width": 768,
            "height": 768
        },
        "class_type": "ModelSamplingFlux",
        "_meta": {"title": "ModelSamplingFlux"}
    },
    "7": {
        "inputs": {
            "seed": 748260172255095,
            "steps": 8,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["6", 0],
            "positive": ["5", 0],
            "negative": ["5", 1],
            "latent_image": ["8", 0]
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"}
    },
    "8": {
        "inputs": {
            "pixels": ["4", 0],
            "vae": ["3", 0]
        },
        "class_type": "VAE Encode",
        "_meta": {"title": "VAE Encode"}
    },
    "9": {
        "inputs": {
            "samples": ["7", 0],
            "vae": ["3", 0]
        },
        "class_type": "VAE Decode",
        "_meta": {"title": "VAE Decode"}
    },
    "10": {
        "inputs": {
            "filename_prefix": "qwen_yiyi_edit",
            "images": ["9", 0]
        },
        "class_type": "Save Image",
        "_meta": {"title": "Save Image"}
    }
}


def run_workflow(input_image_path, prompt_text, seed=None, steps=8, cfg=1.0):
    """
    Run the ComfyUI workflow with custom parameters
    
    Args:
        input_image_path: Path to input image
        prompt_text: Text prompt for image editing
        seed: Random seed (None for random)
        steps: Number of inference steps
        cfg: CFG scale
    """
    # Upload image (assumes image is in ComfyUI input folder)
    image_filename = upload_image(input_image_path)
    
    # Update workflow with parameters
    workflow["4"]["inputs"]["image"] = image_filename
    workflow["5"]["inputs"]["text"] = prompt_text
    workflow["7"]["inputs"]["steps"] = steps
    workflow["7"]["inputs"]["cfg"] = cfg
    
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    workflow["7"]["inputs"]["seed"] = seed
    
    # Queue the prompt
    print(f"Queuing workflow with seed: {seed}")
    result = queue_prompt(workflow, CLIENT_ID)
    prompt_id = result['prompt_id']
    
    print(f"Prompt queued with ID: {prompt_id}")
    print("Processing... (check ComfyUI interface for progress)")
    
    return prompt_id


if __name__ == "__main__":
    # Example usage
    input_image = "input.png"  # Place your image in ComfyUI's input folder
    prompt = "make green eyes WHEN GREEN ONLY on the head to face"
    
    # Make sure your input image exists
    if not os.path.exists(input_image):
        print(f"Warning: {input_image} not found. Please place it in ComfyUI's input folder.")
        print("Creating a sample workflow JSON file instead...")
        
        # Save workflow as JSON for manual import
        with open("workflow.json", "w") as f:
            json.dump(workflow, f, indent=2)
        print("Workflow saved to workflow.json - you can import this in ComfyUI")
    else:
        # Run the workflow
        prompt_id = run_workflow(
            input_image_path=input_image,
            prompt_text=prompt,
            seed=748260172255095,  # Use specific seed or None for random
            steps=8,
            cfg=1.0
        )
        
        print(f"\nWorkflow submitted! Prompt ID: {prompt_id}")
        print("Check ComfyUI interface for results.")
