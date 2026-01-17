import asyncio
import time
import modal
import os
import subprocess
import glob
import shutil
from utils.prompt_generator import generate_prompt_according_image
from utils.extract import extract_json
from utils.generate_single_image import ImageGenerator, app as app_qwen
from utils.generate_stability_diffusion import StabilityDiffusionXLImageGenerator, app as app_sdxl
from constants import ASPECT_RATIO

app = modal.App("new-orchestrator")

app.include(app_qwen)
app.include(app_sdxl)

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("fastapi", "openai", "python-dotenv", "gallery-dl")
    .add_local_dir("utils", remote_path="/root/utils", copy=True)
    .add_local_file("constants.py", remote_path="/root/constants.py", copy=True)
)

def download_image(url: str, output_dir: str = "/tmp/downloads") -> str:
    """
    Downloads an image from the given URL using gallery-dl.
    Returns the path to the downloaded file.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"⬇️ Downloading image from {url}...")
    try:
        subprocess.run(
            ["gallery-dl", "--directory", output_dir, url],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"Error downloading image: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to download image from {url}")

    files = glob.glob(f"{output_dir}/**/*", recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    
    if not files:
        raise FileNotFoundError("No image found after download.")
    
    return files[0]

@app.function(image=image, secrets=[modal.Secret.from_dotenv()])
@modal.fastapi_endpoint(method="POST")
def generate_art(url: str, mode: str = "pfp"):
    """
    Generate art based on an input URL (e.g. Pinterest).
    
    Args:
        url (str): The URL of the reference image.
        mode (str): "pfp" for Profile Pictures (uses Qwen) or "mobile" for Wallpapers (uses SDXL).
    """
    start_time = time.perf_counter()
    
    try:
        downloaded_path = download_image(url)
        print(f"✅ Image downloaded to: {downloaded_path}")
    except Exception as e:
        return {"error": str(e)}

    with open(downloaded_path, "rb") as f:
        image_bytes = f.read()

    print(f"🧠 Generating prompt variants for mode: {mode}...")
    llm_start = time.perf_counter()
    
    prompt_type = "profile" if mode.lower() == "pfp" else "wallpaper"
    prompt_json_str = asyncio.run(generate_prompt_according_image(image_bytes, type=prompt_type))
    prompt_data = extract_json(prompt_json_str)
    print("Prompt Data:", prompt_data)
    llm_duration = time.perf_counter() - llm_start

    if mode.lower() == "pfp":
        target_ratio = ASPECT_RATIO["1:1"]
        generator_class = ImageGenerator
        generator_name = "Qwen (PFP)"
    else: 
        target_ratio = ASPECT_RATIO["9:16"]
        generator_class = StabilityDiffusionXLImageGenerator
        generator_name = "SDXL (Mobile Wallpaper)"

    map_inputs = [
        (target_ratio, variant["prompt"]) 
        for variant in prompt_data["variants"]
    ]

    gpu_start = time.perf_counter()
    print(f"🎨 Generating images with {generator_name}...")
    
    results = list(generator_class().generate.map(map_inputs))
    
    gpu_duration = time.perf_counter() - gpu_start

    total_duration = time.perf_counter() - start_time
    print("\n" + "="*40)
    print(f"🏁 GENERATION COMPLETE ({mode})")
    print(f"⏱️  LLM Prompting: {llm_duration:.2f}s")
    print(f"⏱️  GPU Run ({generator_name}): {gpu_duration:.2f}s")
    print(f"⏱️  Total Pipeline: {total_duration:.2f}s")
    print(f"📈 Output: {results}")
    print("="*40)
    
    return {"mode": mode, "generator": generator_name, "results": results}
