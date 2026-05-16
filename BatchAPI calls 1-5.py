import os
import re
import json
import base64
import pandas as pd
import math
from PIL import Image
import io

EXCEL_PATH = r"D:\Bsc Thesis\Data\Thesis Photos Descriptions.xlsx"
IMAGES_FOLDER = r"D:\Bsc Thesis\Data\Pictures for Thesis"

MODEL_TAG = "Gemini"    # Options: "Qwen", "GPT", "Gemini"
LANG_TAG = "EN"         # Options: "EN", "RO"
STRATEGY_TAG = "S1"
MAX_TOKENS = 10
NUM_CHUNKS = 1     # Increase to split into multiple files (e.g. 6 for Gemini 100mb limit)
STRATEGY_FOLDER = STRATEGY_TAG[:2]
#sample_size = 10       

if MODEL_TAG == "Qwen":
    OUTPUT_FOLDER = os.path.join(r"D:\Bsc Thesis\batch_input\TogetherAI", STRATEGY_TAG)
elif MODEL_TAG == "GPT":
    OUTPUT_FOLDER = os.path.join(r"D:\Bsc Thesis\batch_input\OpenAI", STRATEGY_TAG)
else:
    OUTPUT_FOLDER = os.path.join(r"D:\Bsc Thesis\batch_input\Gemini", STRATEGY_TAG)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def strip_exif_and_encode(img_path):
    """Strip EXIF metadata and return base64-encoded image bytes."""
    with Image.open(img_path) as img:
        clean = img.copy()
        buffer = io.BytesIO()
        clean.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


def build_prompt(context):
    if LANG_TAG == "EN":
        return """Determine if the attached image is AI-generated, AI-altered or Deepfake.
Required strict format:
Verdict: [Yes/No]"""
 
    else:  # RO
        return """Determină dacă fotografia atașată acestei conversații este generată cu ajutorul inteligenței artificiale, este modificată cu ajutorul inteligenței artificiale sau reprezintă un deepfake.
Formatul pentru răspuns:
Verdict: [Da/Nu]"""

def create_batch_files():
    df = pd.read_excel(EXCEL_PATH)

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    #if sample_size:
       #df_shuffled = df_shuffled.head(sample_size)

    total_images = len(df_shuffled)
    chunk_size = math.ceil(total_images / NUM_CHUNKS)
    all_payloads = []

    print(f" Processing {total_images} images for {MODEL_TAG} | {LANG_TAG} | {STRATEGY_TAG}...")

    for index, row in df_shuffled.iterrows():
        raw_val = str(row.iloc[0]).strip()
        number_prefix = re.split(r'\.', raw_val)[0]
        context = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else "No context available."

        pattern = re.compile(rf"^{number_prefix}\..*", re.IGNORECASE)
        image_name = next((fn for fn in os.listdir(IMAGES_FOLDER) if pattern.match(fn)), None)

        if not image_name:
            print(f"Image not found for ID: {number_prefix}, skipping.")
            continue

        img_path = os.path.join(IMAGES_FOLDER, image_name)
        b64_image = strip_exif_and_encode(img_path)

        prompt_text = build_prompt(context)

        # build file and environment structure
        if MODEL_TAG == "GPT":
            payload = {
                "custom_id": f"id-{number_prefix}-{LANG_TAG}-{STRATEGY_TAG}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "auto"}}
                            ]
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": MAX_TOKENS
                }
            }
        elif MODEL_TAG == "Qwen":
            payload = {
                "custom_id": f"id-{number_prefix}-{LANG_TAG}-{STRATEGY_TAG}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "Qwen/Qwen3.5-9B",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                            ]
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": MAX_TOKENS,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            }
        else:  # Gemini
            payload = {
                "custom_id": f"id-{number_prefix}-{LANG_TAG}-{STRATEGY_TAG}",
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": prompt_text},
                                {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}}
                            ]
                        }
                    ],
                    "generation_config": {
                        "temperature": 0,
                        "max_output_tokens": MAX_TOKENS
                    }
                }
            }

        all_payloads.append(payload)

    for i in range(NUM_CHUNKS):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_images)
        chunk = all_payloads[start_idx:end_idx]

        file_name = f"{MODEL_TAG}_{LANG_TAG}_{STRATEGY_TAG}_Split{i+1}.jsonl"
        file_path = os.path.join(OUTPUT_FOLDER, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            for item in chunk:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Created: {file_name} ({len(chunk)} images)")


if __name__ == "__main__":
    create_batch_files()