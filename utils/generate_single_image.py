import time
import modal
import os

# Volumes & constants
model_cache_vol = modal.Volume.from_name("model-cache-vol", create_if_missing=True)

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
        "sentencepiece",
        "boto3",
    )
)

app = modal.App("image-to-image-gen")

@app.cls(
    image=image,
    gpu="H100",
    volumes={
        CACHE_DIR: model_cache_vol,
    },
    secrets=[modal.Secret.from_dotenv()],
    timeout=1200,
    env={
        "HF_HOME": CACHE_DIR,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    },
    max_containers=1,
    scaledown_window=30,
)
class ImageGenerator:
    @modal.enter()
    def load_model(self):
        """Load model once per container startup."""
        from diffusers import DiffusionPipeline
        import torch
        import boto3

        print("🧠 Loading Qwen/Qwen-Image model...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        ).to("cuda")

        self.pipe.enable_attention_slicing()

        print("Model loaded and ready on GPU.")

        try:
            self.s3_client = boto3.client(
                "s3",
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            )
            self.bucket_name = os.environ.get("AWS_BUCKET_NAME")
            
            if self.bucket_name:
                print(f"S3 Configured. Bucket: {self.bucket_name}")
            else:
                print("AWS_BUCKET_NAME not found in environment variables!")
                
        except Exception as e:
            print(f"Failed to init S3: {e}")
            self.s3_client = None
            self.bucket_name = None

    @modal.method()
    def generate(self, args):
        """Generate one image per call, reusing the pre-loaded model."""
        aspect_ratio, prompt_input, category = args
        width, height = aspect_ratio

        positive_magic = ", Ultra HD, 4K, cinematic composition."
        full_prompt = prompt_input + positive_magic
        negative_prompt = (
            "low quality, worst quality, blurry, pixelated, jpeg artifacts, overexposed, underexposed, "
            "bad anatomy, bad proportions, deformed body, extra limbs, missing limbs, "
            "extra fingers, fused fingers, malformed hands, broken pose, incorrect perspective, "
            "duplicate character, multiple faces, crossed eyes, "
            "text, watermark, logo, signature, UI elements, border, frame, "
            "concept art, cinematic, epic, fantasy art, surreal, dreamlike, ethereal, otherworldly, "
            "ai generated, artificial, stylized, illustration, digital painting, matte painting, "
            "neon, glowing, glow effects, bloom, light trails, rainbow colors, psychedelic, "
            "oversaturated, ultra vibrant, color explosion, iridescent, vaporwave, synthwave, cyberpunk, "
            "3d render, unreal engine, octane render, blender render, cgi, vfx, hyperrealistic render, "
            "hdr, excessive contrast, dramatic lighting, rim light, volumetric lighting, god rays"
        )

        print(f"🚀 [PID {os.getpid()}] Generating: '{prompt_input[:40]}...' (Category: {category})")

        # Inference
        from diffusers import DiffusionPipeline
        import torch
        from PIL import Image

        output = self.pipe(
            prompt=full_prompt,
            width=width,
            negative_prompt=negative_prompt,
            height=height,
            num_inference_steps=20,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        # Ensure output directory exists
        output_dir = "/tmp/generated_images"
        os.makedirs(output_dir, exist_ok=True)

        # Save image
        timestamp = int(time.time())
        filename = f"premium_{timestamp}_{os.getpid()}.png"
        file_path = os.path.join(output_dir, filename)
        output.save(file_path)

        # Upload to S3
        s3_key = None
        if self.s3_client and self.bucket_name:
            # Sane category string
            safe_category = "".join(x for x in str(category) if x.isalnum() or x in "_-").lower()
            if not safe_category:
                safe_category = "uncategorized"
                
            s3_key = f"premium_raw_images/{safe_category}/{filename}"
            try:
                self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
                print(f"Uploaded to S3: {s3_key}")
            except Exception as e:
                print(f"S3 Upload Error: {e}")

        return f"Saved to {file_path}, Uploaded to {s3_key or 'Local Only'}"
