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
        "git+https://github.com/huggingface/diffusers",
        "pillow",
        "torch",
        "transformers",
        "accelerate",
        "sentencepiece"
    )
)

app = modal.App("image-to-image-gen")

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
class ImageGenerator:
    @modal.enter()
    def load_model(self):
        """Load model once per container startup."""
        from diffusers import DiffusionPipeline
        import torch

        print("🧠 Loading Qwen/Qwen-Image model...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        ).to("cuda")

        self.pipe.enable_attention_slicing()

        print("✅ Model loaded and ready on GPU.")

    @modal.method()
    def generate(self, args):
        """Generate one image per call, reusing the pre-loaded model."""
        aspect_ratio, prompt_input = args
        width, height = aspect_ratio

        # Enhance prompt
        positive_magic = ", Ultra HD, 4K, cinematic composition."
        full_prompt = prompt_input + positive_magic

        print(f"🚀 [PID {os.getpid()}] Generating: '{prompt_input[:40]}...'")

        # Inference
        from diffusers import DiffusionPipeline
        import torch
        from PIL import Image

        output = self.pipe(
            prompt=full_prompt,
            width=width,
            negative_prompt = " ",
            height=height,
            num_inference_steps=20,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        # Ensure output directory exists
        os.makedirs(VOLUME_PATH, exist_ok=True)

        # Save image
        timestamp = int(time.time())
        filename = f"premium_{timestamp}_{os.getpid()}.png"
        file_path = os.path.join(VOLUME_PATH, filename)
        output.save(file_path)

        # Persist to volume
        data_vol.commit()

        return f"Saved to {file_path}"

