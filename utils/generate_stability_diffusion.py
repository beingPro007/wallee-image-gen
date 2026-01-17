import time
import modal
import os

# Volumes & constants
model_cache_vol = modal.Volume.from_name("model-cache-vol", create_if_missing=True)
data_vol = modal.Volume.from_name("my-image-storage", create_if_missing=True)
CACHE_DIR = "/cache"
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "diffusers>=0.21.0",
        "pillow",
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "sentencepiece"
    )
)

app = modal.App("sdxl-image-gen")

@app.cls(
    image=image,
    gpu="H100",
    volumes={
        CACHE_DIR: model_cache_vol,
        VOLUME_PATH: data_vol,
    },
    timeout=1200,
    env={
        "HF_HOME": CACHE_DIR,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    },
    max_containers=1,
)
class StabilityDiffusionXLImageGenerator:
    @modal.enter()
    def load_model(self):
        """Load SDXL 1.0 once per container startup."""
        from diffusers import DiffusionPipeline
        import torch

        print("🧠 Loading Stable Diffusion XL 1.0...")
        
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir=CACHE_DIR,
        ).to("cuda")

        self.pipe.enable_attention_slicing()
        
        print("✅ SDXL loaded and ready on GPU.")

    @modal.method()
    def generate(self, args):
        """Generate image using SDXL."""
        aspect_ratio, prompt_input = args
        width, height = aspect_ratio

        positive_magic = ", highly detailed, photorealistic, 8k resolution, cinematic lighting"
        full_prompt = prompt_input + positive_magic
        
        negative_prompt = "artistic, drawing, painting, sketch, cartoon, anime, blurry, low quality, distorted"

        print(f"🚀 [PID {os.getpid()}] Generating with SDXL: '{prompt_input[:40]}...'")

        import torch
        
        output = self.pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=30, 
            guidance_scale=7.5,    
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        os.makedirs(VOLUME_PATH, exist_ok=True)

        # Save image
        timestamp = int(time.time())
        filename = f"sdxl_{timestamp}_{os.getpid()}.png"
        file_path = os.path.join(VOLUME_PATH, filename)
        output.save(file_path)

        data_vol.commit()

        return f"Saved to {file_path}"