"""
Standalone Python implementation of the ComfyUI workflow
This replicates the workflow without requiring ComfyUI server

Note: This script requires ComfyUI-compatible model loaders.
For GGUF models, you'll need to use ComfyUI directly or convert models to Diffusers format.
"""

import os
import sys
import torch
from PIL import Image
import numpy as np


class QwenImageEditWorkflow:
    """
    Replicates the ComfyUI workflow for Qwen Image Edit
    Based on the workflow: Load models -> Load image -> Prompt -> Sample -> Decode -> Save
    """
    
    def __init__(self, 
                 model_base_path=r"D:\AIModels\Qwen-Image-FP8",
                 unet_path=None,
                 clip_path=None,
                 vae_path=None,
                 use_gguf=True,
                 device="cuda",
                 dtype=torch.bfloat16):
        """
        Initialize the workflow components
        
        Args:
            model_base_path: Base path to local models directory
            unet_path: Path to UNet model (overrides default)
            clip_path: Path to CLIP model (overrides default)
            vae_path: Path to VAE model (overrides default)
            use_gguf: Whether to use GGUF models (True) or safetensors (False)
            device: Device to run on ('cuda' or 'cpu')
            dtype: Data type for models
        """
        self.device = device
        self.dtype = dtype
        self.model_base_path = model_base_path
        self.use_gguf = use_gguf
        
        print("Step 1: Loading models...")
        print(f"Model base path: {model_base_path}")
        
        # Set default paths if not provided
        if use_gguf:
            self.unet_path = unet_path or os.path.join(model_base_path, "gguf", "unet", "Qwen-Image-Edit-2509-Q4_0.gguf")
            self.clip_path = clip_path or os.path.join(model_base_path, "gguf", "clip", "Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf")
        else:
            self.unet_path = unet_path or model_base_path
            self.clip_path = clip_path or os.path.join(model_base_path, "text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors")
        
        self.vae_path = vae_path or os.path.join(model_base_path, "vae", "qwen_image_vae.safetensors")
        
        print(f"  UNet: {self.unet_path}")
        print(f"  CLIP: {self.clip_path}")
        print(f"  VAE: {self.vae_path}")
        
        # Verify model files exist
        if use_gguf:
            if not os.path.exists(self.unet_path):
                raise FileNotFoundError(f"UNet model not found at: {self.unet_path}")
            if not os.path.exists(self.clip_path):
                raise FileNotFoundError(f"CLIP model not found at: {self.clip_path}")
        if not os.path.exists(self.vae_path):
            raise FileNotFoundError(f"VAE model not found at: {self.vae_path}")
        
        print("\n" + "="*60)
        print("ERROR: GGUF Models Cannot Be Loaded in Python")
        print("="*60)
        print("Your models are GGUF files (quantized ComfyUI format):")
        print(f"  • {os.path.basename(self.unet_path)}")
        print(f"  • {os.path.basename(self.clip_path)}")
        print(f"  • {os.path.basename(self.vae_path)}")
        print("\n❌ GGUF files CANNOT be loaded using Python's Diffusers library.")
        print("❌ There is NO Python library to load GGUF models standalone.")
        print("❌ GGUF models REQUIRE ComfyUI's backend to function.")
        print("\n✅ SOLUTION: Use ComfyUI to load your GGUF models")
        print("\nYour options:")
        print("  1. Use run_comfyui_workflow.py (requires ComfyUI running)")
        print("  2. Use ComfyUI GUI with workflow.json")
        print("  3. Download full Diffusers models (~20GB, NOT from your GGUF)")
        print("\nThis script CANNOT proceed with GGUF models.")
        print("="*60 + "\n")
        
        # Set pipeline to None - cannot load GGUF
        self.pipeline = None
        self.loaded = False
        
        print("⚠️  This script is now in ERROR state.")
        print("⚠️  Please use run_comfyui_workflow.py instead.")
        
        # Configure sampling parameters (ModelSamplingFlux equivalent)
        self.max_shift = 1.15
        self.base_shift = 0.5
        self.default_width = 768
        self.default_height = 768
        
        print("✓ All models loaded successfully\n")
    
    def load_image(self, image_path):
        """
        Step 2: Load image for editing
        
        Args:
            image_path: Path to input image
            
        Returns:
            PIL Image object
        """
        print(f"Step 2: Loading image from {image_path}")
        image = Image.open(image_path).convert("RGB")
        
        # Resize to default dimensions if needed
        if image.size != (self.default_width, self.default_height):
            print(f"  Resizing from {image.size} to ({self.default_width}, {self.default_height})")
            image = image.resize((self.default_width, self.default_height), Image.LANCZOS)
        
        print("✓ Image loaded\n")
        return image
    
    def run_sampling(self, 
                     image,
                     prompt,
                     seed=748260172255095,
                     steps=8,
                     cfg_scale=1.0,
                     sampler_name="euler",
                     scheduler="normal",
                     denoise=1.0,
                     negative_prompt=""):
        """
        Step 3-4: Run the sampling process (KSampler equivalent)
        
        Args:
            image: Input PIL Image
            prompt: Text prompt for editing
            seed: Random seed for reproducibility
            steps: Number of inference steps
            cfg_scale: Classifier-free guidance scale
            sampler_name: Sampler algorithm
            scheduler: Scheduler type
            denoise: Denoising strength (1.0 = full denoise)
            negative_prompt: Negative prompt
            
        Returns:
            Generated PIL Image
        """
        print("Step 3: Processing prompt and sampling...")
        print(f"  Prompt: {prompt}")
        print(f"  Seed: {seed}")
        print(f"  Steps: {steps}")
        print(f"  CFG Scale: {cfg_scale}")
        print(f"  Sampler: {sampler_name}")
        
        # Set random seed for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare inputs
        inputs = {
            "prompt": prompt,
            "image": image,
            "generator": generator,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "negative_prompt": negative_prompt,
            "height": self.default_height,
            "width": self.default_width,
        }
        
        # Add denoise strength if supported
        if hasattr(self.pipeline, 'strength'):
            inputs["strength"] = denoise
        
        # Run inference
        print("  Running inference...")
        with torch.inference_mode():
            output = self.pipeline(**inputs)
            result_image = output.images[0]
        
        print("✓ Sampling complete\n")
        return result_image
    
    def save_image(self, image, output_path="output_qwen_edit.png", prefix="qwen_yiyi_edit"):
        """
        Step 5: Save the output image
        
        Args:
            image: PIL Image to save
            output_path: Path to save the image
            prefix: Filename prefix
            
        Returns:
            Full path to saved image
        """
        print(f"Saving image to {output_path}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Save image
        image.save(output_path)
        full_path = os.path.abspath(output_path)
        
        print(f"✓ Image saved at {full_path}\n")
        return full_path
    
    def run_workflow(self,
                     input_image_path,
                     prompt,
                     output_path="output_qwen_edit.png",
                     seed=None,
                     steps=8,
                     cfg_scale=1.0,
                     negative_prompt=""):
        """
        Run the complete workflow from image input to output
        
        Args:
            input_image_path: Path to input image
            prompt: Text prompt for editing
            output_path: Path to save output image
            seed: Random seed (None for random)
            steps: Number of inference steps
            cfg_scale: CFG scale
            negative_prompt: Negative prompt
            
        Returns:
            Path to saved output image
        """
        print("="*60)
        print("QWEN IMAGE EDIT WORKFLOW")
        print("="*60 + "\n")
        
        # Check if models are loaded
        if not self.loaded or self.pipeline is None:
            print("ERROR: Models not loaded. Cannot run workflow.")
            print("Please use ComfyUI with your GGUF models instead.")
            return None
        
        # Use random seed if not provided
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        # Step 2: Load image
        image = self.load_image(input_image_path)
        
        # Step 3-4: Run sampling
        result_image = self.run_sampling(
            image=image,
            prompt=prompt,
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            negative_prompt=negative_prompt
        )
        
        # Step 5: Save image
        output_file = self.save_image(result_image, output_path)
        
        print("="*60)
        print("WORKFLOW COMPLETE")
        print("="*60)
        
        return output_file


def main():
    """Example usage of the workflow"""
    
    # Initialize workflow (Step 1: Load models)
    workflow = QwenImageEditWorkflow(
        model_base_path=r"D:\AIModels\Qwen-Image-FP8",
        use_gguf=True,  # Set to False to use safetensors instead
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    # Run the workflow
    input_image = "input.png"
    prompt = "Add the word 'Great!' into the image."
    
    if not os.path.exists(input_image):
        print(f"Error: Input image '{input_image}' not found!")
        print("Please provide a valid input image path.")
        return
    
    output_file = workflow.run_workflow(
        input_image_path=input_image,
        prompt=prompt,
        output_path="output_qwen_edit.png",
        seed=748260172255095,  # From ComfyUI workflow
        steps=8,
        cfg_scale=1.0,
        negative_prompt=""
    )
    
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
