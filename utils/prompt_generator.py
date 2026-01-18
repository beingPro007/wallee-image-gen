import base64
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI()

async def generate_prompt_according_image(
    image: bytes,
    count: str,
    type: str = "Regular",
    user_input: str | None = None,
    style: str | None = None
) -> str:
    image_base64 = base64.b64encode(image).decode("utf-8")

    if type.lower() == "profile":
        format_instructions = (
            "- FORMAT: Circular Profile Picture.\n"
            "- CRITICAL: Subject centered. Outside circle must be solid black.\n"
        )
    elif type.lower() == "desktop":
        format_instructions = (
            "- FORMAT: 16:9 Desktop Wallpaper.\n"
            "- COMPOSITION: Wide cinematic framing with side negative space.\n"
        )
    else:
        format_instructions = (
            "- FORMAT: 9:16 Vertical Mobile Wallpaper.\n"
            "- COMPOSITION: Tall, full-frame vertical shot.\n"
        )

    user_context = ""
    if user_input and user_input.strip():
        user_context += f"USER INSTRUCTION (PRIORITY #1): {user_input}\n"
    else:
        user_context += "USER INSTRUCTION: None. rely on image analysis.\n"

    if style and style.strip():
        user_context += f"REQUIRED ART STYLE: {style}\n"
    else:
        user_context += "REQUIRED ART STYLE: DIVERSE (Generate a distinct art style for each variant).\n"

    prompt_content = (
        f"Step 1: IDENTIFY the specific character, celebrity, or subject. "
        f"Look for anime characters (e.g. Toji Fushiguro, Goku, Naruto) or public figures. "
        f"You MUST use the exact name and source material (e.g. 'from Jujutsu Kaisen') in every prompt.\n\n"

        f"Step 2: INTERNALIZE the User Instructions:\n"
        f"{user_context}\n"
        f"   - If the user specifies a concept or character, it overrides visual analysis.\n"
        f"   - If the user specifies a style, ALL variants must use that style but differ in composition/pose.\n"
        f"   - If no style is specified, every variant MUST be a completely different medium (e.g. 1. 3D Render, 2. Oil Painting, 3. Anime, 4. Photorealistic).\n\n"

        f"Step 3: Generate {count} variants.\n"
        f"   - Focus strictly on the SUBJECT IDENTITY defined in Step 1 or User Input.\n"
        f"   - Ensure the subject is the central focus.\n"
        f"   - Avoid generic descriptions like 'man with scar' if you know it is 'Toji Fushiguro'.\n\n"

        f"Step 4: Categorize the image.\n"
        f"   - specificy a single folder name (lowercase, no spaces, use underscores) that fits the image niche.\n"
        f"   - Examples: 'anime_characters', 'cars', 'landscapes', 'portraits', 'fantasy', 'abstract', 'animals'.\n"
        f"   - If the image is a known character, use the series name or 'anime_characters'.\n\n"

        f"Step 5: Prompt Formatting Rules:\n"
        f"{format_instructions}\n"
        f"   - STRICTLY under 40 words per prompt.\n"
        f"   - Comma-separated descriptive tags only.\n"
        f"   - Extremely detailed, cinematic, high quality, 8k.\n"
        f"   - No explanations.\n\n"

        "Return ONLY valid JSON:\n"
        '{"variants": ['
        '{"variant": 1, "category": "category_name", "prompt": "..."}, '
        '{"variant": 2, "category": "category_name", "prompt": "..."}, '
        '{"variant": 3, "category": "category_name", "prompt": "..."}, '
        '{"variant": 4, "category": "category_name", "prompt": "..."}, '
        '{"variant": 5, "category": "category_name", "prompt": "..."}'
        ']}'
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert visual prompt engineer. "
                    "Your goal is to create high-quality, precise image generation prompts. "
                    "You prioritize specific named entities (characters, places) and user instructions above all else."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        ],
        max_tokens=1000,
        response_format={"type": "json_object"}
    )

    message = response.choices[0].message

    if hasattr(message, "parsed") and message.parsed is not None:
        return message.parsed
    elif message.content is not None:
        return message.content
    elif hasattr(message, "refusal") and message.refusal:
        raise RuntimeError(f"LLM Refused to generate prompt: {message.refusal}")
    else:
        finish_reason = response.choices[0].finish_reason
        raise RuntimeError(f"LLM returned no content. Finish Reason: {finish_reason}")

