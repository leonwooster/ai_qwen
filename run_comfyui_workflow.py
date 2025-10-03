"""
ComfyUI Workflow Runner - Uses YOUR GGUF models from the screenshot

This is the CORRECT way to use your GGUF models:
  • Qwen-Image-Edit-2509-Q4_0.gguf (UNet)
  • Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf (CLIP)
  • qwen_image_vae.safetensors (VAE)

This script does NOT download "Qwen/Qwen-Image-Edit-2509" from Hugging Face.
It uses your LOCAL GGUF models through ComfyUI's API.

Prerequisites:
1. ComfyUI must be running (usually at http://127.0.0.1:8188)
2. Your GGUF models must be in ComfyUI's models directory:
   - models/unet/Qwen-Image-Edit-2509-Q4_0.gguf
   - models/clip/Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf
   - models/vae/qwen_image_vae.safetensors
3. Install websocket-client: pip install websocket-client

Usage:
    python run_comfyui_workflow.py
"""

import json
import urllib.request
import urllib.parse
import random
import os
import time
import websocket
import uuid
from PIL import Image
import io


class ComfyUIWorkflowRunner:
    """Run ComfyUI workflows programmatically"""
    
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        
    def queue_prompt(self, prompt):
        """Send a prompt to ComfyUI server"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        req.add_header('Content-Type', 'application/json')
        
        try:
            response = urllib.request.urlopen(req)
            return json.loads(response.read())
        except Exception as e:
            print(f"Error queuing prompt: {e}")
            print(f"Make sure ComfyUI is running at http://{self.server_address}")
            return None
    
    def get_history(self, prompt_id):
        """Get the execution history for a prompt"""
        try:
            with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
                return json.loads(response.read())
        except Exception as e:
            print(f"Error getting history: {e}")
            return None
    
    def get_image(self, filename, subfolder, folder_type):
        """Retrieve generated image from ComfyUI"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        
        try:
            with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
                return response.read()
        except Exception as e:
            print(f"Error retrieving image: {e}")
            return None
    
    def upload_image(self, image_path):
        """Upload image to ComfyUI server"""
        import mimetypes
        from io import BytesIO
        
        # Read the image file
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Prepare multipart form data
        boundary = f'----WebKitFormBoundary{random.randint(1000000000, 9999999999)}'
        body = BytesIO()
        
        # Add image field
        body.write(f'--{boundary}\r\n'.encode())
        body.write(f'Content-Disposition: form-data; name="image"; filename="{os.path.basename(image_path)}"\r\n'.encode())
        body.write(f'Content-Type: {mimetypes.guess_type(image_path)[0] or "image/png"}\r\n\r\n'.encode())
        body.write(image_data)
        body.write(f'\r\n--{boundary}--\r\n'.encode())
        
        # Send request
        req = urllib.request.Request(
            f"http://{self.server_address}/upload/image",
            data=body.getvalue(),
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
        )
        
        try:
            response = urllib.request.urlopen(req)
            result = json.loads(response.read())
            return result.get('name', os.path.basename(image_path))
        except Exception as e:
            print(f"Error uploading image: {e}")
            return os.path.basename(image_path)
    
    def wait_for_completion(self, prompt_id, timeout=300):
        """Wait for workflow to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            
            if history and prompt_id in history:
                # Check if completed
                return history[prompt_id]
            
            time.sleep(1)
            print(".", end="", flush=True)
        
        print("\nTimeout waiting for completion")
        return None
    
    def run_workflow(self, 
                     input_image_path,
                     prompt_text,
                     output_path="output_comfyui.png",
                     seed=None,
                     steps=8,
                     cfg=1.0,
                     width=768,
                     height=768):
        """
        Run the Qwen Image Edit workflow
        
        Args:
            input_image_path: Path to input image
            prompt_text: Text prompt for editing
            output_path: Where to save the output
            seed: Random seed (None for random)
            steps: Number of inference steps
            cfg: CFG scale
            width: Image width
            height: Image height
        """
        print("="*60)
        print("COMFYUI WORKFLOW RUNNER")
        print("="*60)
        print("Using YOUR LOCAL GGUF models (from screenshot):")
        print("  • Qwen-Image-Edit-2509-Q4_0.gguf")
        print("  • Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf")
        print("  • qwen_image_vae.safetensors")
        print(f"\nServer: http://{self.server_address}")
        print(f"Input: {input_image_path}")
        print(f"Prompt: {prompt_text}")
        print(f"Steps: {steps}, CFG: {cfg}, Seed: {seed or 'random'}")
        print("="*60 + "\n")
        
        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Upload image
        print("Uploading image...")
        uploaded_filename = self.upload_image(input_image_path)
        print(f"✓ Image uploaded: {uploaded_filename}")
        
        # Create workflow
        workflow = {
            "1": {
                "inputs": {
                    "unet_name": "Qwen-Image-Edit-2509-Q4_0.gguf",
                    "weight_dtype": "default"
                },
                "class_type": "UnetLoader (GGUF)",
                "_meta": {"title": "UnetLoader (GGUF)"}
            },
            "2": {
                "inputs": {
                    "clip_name": "Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf",
                    "type": "flux"
                },
                "class_type": "CLIPLoader (GGUF)",
                "_meta": {"title": "CLIPLoader (GGUF)"}
            },
            "3": {
                "inputs": {
                    "vae_name": "qwen_image_vae.safetensors"
                },
                "class_type": "Load VAE",
                "_meta": {"title": "Load VAE"}
            },
            "4": {
                "inputs": {
                    "image": uploaded_filename,
                    "upload": "image"
                },
                "class_type": "Load Image",
                "_meta": {"title": "Load Image"}
            },
            "5": {
                "inputs": {
                    "text": prompt_text,
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
                    "width": width,
                    "height": height
                },
                "class_type": "ModelSamplingFlux",
                "_meta": {"title": "ModelSamplingFlux"}
            },
            "7": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
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
                    "filename_prefix": "qwen_output",
                    "images": ["9", 0]
                },
                "class_type": "Save Image",
                "_meta": {"title": "Save Image"}
            }
        }
        
        # Queue the workflow
        print("\nQueuing workflow...")
        result = self.queue_prompt(workflow)
        
        if not result:
            print("✗ Failed to queue workflow")
            return None
        
        prompt_id = result.get('prompt_id')
        print(f"✓ Workflow queued (ID: {prompt_id})")
        
        # Wait for completion
        print("\nWaiting for completion", end="")
        history = self.wait_for_completion(prompt_id)
        print()
        
        if not history:
            print("✗ Workflow did not complete")
            return None
        
        # Get the output images
        print("\nRetrieving output...")
        outputs = history.get('outputs', {})
        
        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                for image_info in node_output['images']:
                    filename = image_info['filename']
                    subfolder = image_info.get('subfolder', '')
                    folder_type = image_info.get('type', 'output')
                    
                    # Download image
                    image_data = self.get_image(filename, subfolder, folder_type)
                    
                    if image_data:
                        # Save image
                        with open(output_path, 'wb') as f:
                            f.write(image_data)
                        
                        print(f"✓ Image saved to: {os.path.abspath(output_path)}")
                        
                        print("\n" + "="*60)
                        print("WORKFLOW COMPLETE")
                        print("="*60)
                        
                        return output_path
        
        print("✗ No output images found")
        return None


def main():
    """Example usage"""
    
    # Initialize runner
    runner = ComfyUIWorkflowRunner(server_address="127.0.0.1:8188")
    
    # Configuration
    input_image = "input.png"
    prompt = "Add the word 'Great!' into the image."
    output_path = "output_comfyui.png"
    
    # Check if input exists
    if not os.path.exists(input_image):
        print(f"Error: Input image '{input_image}' not found!")
        print("Please provide a valid input image.")
        return
    
    # Run workflow
    result = runner.run_workflow(
        input_image_path=input_image,
        prompt_text=prompt,
        output_path=output_path,
        seed=748260172255095,  # Use specific seed or None for random
        steps=8,
        cfg=1.0,
        width=768,
        height=768
    )
    
    if result:
        print(f"\n✓ Success! Output saved to: {result}")
    else:
        print("\n✗ Workflow failed")
        print("\nTroubleshooting:")
        print("1. Make sure ComfyUI is running")
        print("2. Check that all models are in the correct directories:")
        print("   - models/unet/Qwen-Image-Edit-2509-Q4_0.gguf")
        print("   - models/clip/Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf")
        print("   - models/vae/qwen_image_vae.safetensors")
        print("3. Verify ComfyUI has the required custom nodes installed")


if __name__ == "__main__":
    main()
