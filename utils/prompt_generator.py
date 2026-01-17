import base64
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI()

async def generate_prompt_according_image(image: bytes, type: str = "Regular") -> str:
    image_base64 = base64.b64encode(image).decode("utf-8")
    
    # 1. Define instructions based on type
    if type.lower() == "profile":
        style_instructions = (
            "- CRITICAL: The output must be a 'Circular Profile Picture (PFP)'.\n"
            "- Feature the subject inside a perfectly centered circle.\n"
            "- Explicitly state that everything outside this circle is a solid, pure black background (#000000).\n"
            "- Use terms like. 'circular vignette' or 'enclosed in a central orb'."
        )
    else:
        style_instructions = (
            "- Focus on a standard full-frame cinematic rectangular composition.\n"
            "- Describe a rich, detailed background that fills the entire frame.\n"
            "- Do NOT mention circles, orbs, or black borders."
        )

    # 2. Build the final prompt
    prompt_content = (
        f"Analyze this image's style and subject.\n\n"
        f"Generate 1 prompt variations. Each must:\n"
        f"{style_instructions}\n"
        "- Use high-contrast, graphic novel, or digital art styles.\n"
        "- Vary the inner theme (e.g., cybernetic, ethereal, volcanic, crystalline).\n\n"
        "Return ONLY JSON:\n"
        '{"variants": [{"variant": 1, "prompt": "..."}]}'
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a professional prompt engineer. Output valid JSON only."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_content
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        ],
        max_tokens=1000,
        response_format={ "type": "json_object" } # Ensures strict JSON output
    )

    return response.choices[0].message.content