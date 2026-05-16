import os
import re
import json
import base64
import pandas as pd
import math
from PIL import Image
import io

EXCEL_PATH = r"C:\Users\giurg\Desktop\Bsc Thesis\Data\Thesis Photos Descriptions.xlsx"
IMAGES_FOLDER = r"C:\Users\giurg\Desktop\Bsc Thesis\Data\Pictures for Thesis"

MODEL_TAG = "Qwen"      # Options: "Qwen", "GPT", "Gemini"
LANG_TAG = "EN"         # Options: "EN", "RO"
STRATEGY_TAG = "S6"
MAX_TOKENS = 250
NUM_CHUNKS = 15
#sample_size = 10       

if MODEL_TAG == "Qwen":
    OUTPUT_FOLDER = os.path.join(r"C:\Users\giurg\Desktop\Bsc Thesis\batch_input\TogetherAI", STRATEGY_TAG)
elif MODEL_TAG == "GPT":
    OUTPUT_FOLDER = os.path.join(r"C:\Users\giurg\Desktop\Bsc Thesis\batch_input\OpenAI", STRATEGY_TAG)
else:
    OUTPUT_FOLDER = os.path.join(r"C:\Users\giurg\Desktop\Bsc Thesis\batch_input\Gemini", STRATEGY_TAG)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def strip_exif_and_encode(img_path):
    with Image.open(img_path) as img:
        clean = img.copy()
        buffer = io.BytesIO()
        clean.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


CONTEXT_23  = "Vasile-al meu!"
CONTEXT_640 = "Am convocat astăzi ședința CSAT pentru analiza evoluției situației din Orientul Mijlociu și impactul economic al războiului, în special creșterea prețului petrolului pe piețele internaționale. Una dintre preocupările noastre importante este situația românilor din zonă. Până la acest moment, au revenit în țară aproximativ 5.700 de oameni, dintre care 3.700 au fost ajutați într-un fel sau altul de autorități, prin munca oamenilor din MAE, pe care o salut."
CONTEXT_208 = "Dacă el eu este România ! Noi trebuie să-l susținem cu orice preț !"
CONTEXT_705 = "O administrație tânără reprezintă viitorul. Tinerii nu duc în spate păcatele corupției care a măcinat această țară. Vom Reface România printr-o nouă generație de politicieni, pentru o nouă generație de români demni și mândri! 📷 Ilie Bumbac"

EXAMPLE_ORDER    = [23, 640, 208, 705]
EXAMPLE_CONTEXTS = {23: CONTEXT_23, 640: CONTEXT_640, 208: CONTEXT_208, 705: CONTEXT_705}

ANSWERS_EN = {
    23:  "1. Image decision: Yes\n2. Detected cues: b,c,d\n3. Political propaganda/manipulation: Yes\n4. Manipulation technique: 2,5\n5. Political preference: Right",
    640: "1. Image decision: No\n2. Detected cues: N\n3. Political propaganda/manipulation: No\n4. Manipulation technique: N\n5. Political preference: Neutral",
    208: "1. Image decision: Yes\n2. Detected cues: a,c,d,e\n3. Political propaganda/manipulation: Yes\n4. Manipulation technique: 1,4\n5. Political preference: Right",
    705: "1. Image decision: No\n2. Detected cues: N\n3. Political propaganda/manipulation: Yes\n4. Manipulation technique: 1,5\n5. Political preference: Right",
}
ANSWERS_RO = {
    23:  "1. Decizie imagine: Da\n2. Indicii detectate: b,c,d\n3. Propagandă politică/manipulare: Da\n4. Tehnică de manipulare: 2,5\n5. Preferință politică: Dreapta",
    640: "1. Decizie imagine: Nu\n2. Indicii detectate: N\n3. Propagandă politică/manipulare: Nu\n4. Tehnică de manipulare: N\n5. Preferință politică: Neutru",
    208: "1. Decizie imagine: Da\n2. Indicii detectate: a,c,d,e\n3. Propagandă politică/manipulare: Da\n4. Tehnică de manipulare: 1,4\n5. Preferință politică: Dreapta",
    705: "1. Decizie imagine: Nu\n2. Indicii detectate: N\n3. Propagandă politică/manipulare: Da\n4. Tehnică de manipulare: 1,5\n5. Preferință politică: Dreapta",
}


def load_example_images():
    b64 = {}
    for img_id in EXAMPLE_ORDER:
        pattern = re.compile(rf"^{img_id}\..*", re.IGNORECASE)
        match = next((fn for fn in os.listdir(IMAGES_FOLDER) if pattern.match(fn)), None)
        if not match:
            raise FileNotFoundError(f"S6 example image {img_id} not found in {IMAGES_FOLDER}")
        b64[img_id] = strip_exif_and_encode(os.path.join(IMAGES_FOLDER, match))
        print(f" Loaded example image {img_id}")
    return b64


def build_s6_content_openai(context, b64_query, examples_b64):
    answers = ANSWERS_EN if LANG_TAG == "EN" else ANSWERS_RO
    if LANG_TAG == "EN":
        intro      = "You will be shown 4 examples of images with their associated social media context and the correct analysis. Study the pattern, then analyze the final image in the exact same format."
        ex_label   = "Example"
        query_lead = f"Now analyze the following image using the exact same format.\nContext: {context}"
        fmt        = ("Answer each field with the required code only. Do not include any additional text or notes. Respond with necessary codes only, no extra text.\n"
                      "Required strict answering format:\n"
                      "1. Image decision: [Yes/No] (Is the image AI-generated/Deepfake/AI-altered?)\n"
                      "2. Reasoning — detected cues: [Select ALL that apply, comma-separated: a/b/c/d/e/N] a. Anatomical implausibility / b. Scene incoherence / c. Contextual implausibility / d. Digital artifacts / e. Textual errors in image / N. Real image (no cues detected). Select N only if image decision is No.\n"
                      "3. Political propaganda/manipulation: [Yes/No]\n"
                      "4. Manipulation/Propaganda technique: [Select ALL that apply, comma-separated: 1/2/3/4/5/N] (1. Emotional appeal / 2. Name-calling or smear / 3. False context or decontextualization / 4. Appeal to authority / 5. Simplification / N. Not applicable). Select N if field 3 is No.\n"
                      "5. Political preference: [Left/Right/Neutral]")
    else:
        intro      = "Vei vedea 4 exemple de imagini cu contextul asociat de pe rețelele sociale și analiza corectă. Studiază structura, apoi analizează imaginea finală în același format."
        ex_label   = "Exemplu"
        query_lead = f"Acum analizează imaginea următoare folosind același format.\nContext: {context}"
        fmt        = ("Răspunde la fiecare câmp doar cu codul cerut. Nu include niciun text suplimentar sau o notă analitică. Răspunde doar cu codurile necesare, fără text suplimentar.\n"
                      "Format strict de răspuns obligatoriu:\n"
                      "1. Decizie imagine: [Da/Nu] (Imaginea este generată de AI / Deepfake / modificată cu AI?)\n"
                      "2. Raționament — indicii detectate: [Selectează TOATE cele aplicabile, separate prin virgulă: a/b/c/d/e/N] a. Implausibilitate anatomică / b. Incoerență de scenă / c. Implausibilitate contextuală / d. Artefacte digitale / e. Erori textuale în imagine / N. Imagine reală. Selectează N doar dacă decizia imaginii este Nu.\n"
                      "3. Propagandă politică/manipulare: [Da/Nu]\n"
                      "4. Tehnică de manipulare/propagandă: [Selectează TOATE cele aplicabile, separate prin virgulă: 1/2/3/4/5/N] (1. Apel emoțional / 2. Etichetare negativă sau defăimare / 3. Context fals sau decontextualizare / 4. Apel la autoritate / 5. Simplificare / N. Neaplicabil). Selectează N dacă câmpul 3 este Nu.\n"
                      "5. Preferință politică: [Stânga/Dreapta/Neutru]")

    content = [{"type": "text", "text": intro}]
    for i, img_id in enumerate(EXAMPLE_ORDER, start=1):
        content.append({"type": "text", "text": f"{ex_label} {i} — context: {EXAMPLE_CONTEXTS[img_id]}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{examples_b64[img_id]}"}})
        content.append({"type": "text", "text": answers[img_id]})
    content.append({"type": "text", "text": query_lead})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_query}"}})
    content.append({"type": "text", "text": fmt})
    return content


def build_s6_content_gemini(context, b64_query, examples_b64):
    answers = ANSWERS_EN if LANG_TAG == "EN" else ANSWERS_RO
    if LANG_TAG == "EN":
        intro      = "You will be shown 4 examples of images with their associated social media context and the correct analysis. Study the pattern, then analyze the final image in the exact same format."
        ex_label   = "Example"
        query_lead = f"Now analyze the following image using the exact same format.\nContext: {context}"
        fmt        = "Answer each field with the required code only. Do not include any additional text or notes.\nRequired strict answering format:\n1. Image decision: [Yes/No]\n2. Detected cues: [a/b/c/d/e/N, comma-separated]\n3. Political propaganda/manipulation: [Yes/No]\n4. Manipulation technique: [1/2/3/4/5/N, comma-separated]\n5. Political preference: [Left/Right/Neutral]"
    else:
        intro      = "Vei vedea 4 exemple de imagini cu contextul asociat de pe rețelele sociale și analiza corectă. Studiază structura, apoi analizează imaginea finală în același format."
        ex_label   = "Exemplu"
        query_lead = f"Acum analizează imaginea următoare folosind același format.\nContext: {context}"
        fmt        = "Răspunde la fiecare câmp doar cu codul cerut. Nu include niciun text suplimentar sau o notă analitică.\nFormat strict de răspuns obligatoriu:\n1. Decizie imagine: [Da/Nu]\n2. Indicii detectate: [a/b/c/d/e/N, separate prin virgulă]\n3. Propagandă politică/manipulare: [Da/Nu]\n4. Tehnică de manipulare: [1/2/3/4/5/N, separate prin virgulă]\n5. Preferință politică: [Stânga/Dreapta/Neutru]"

    parts = [{"text": intro}]
    for i, img_id in enumerate(EXAMPLE_ORDER, start=1):
        parts.append({"text": f"{ex_label} {i} — context: {EXAMPLE_CONTEXTS[img_id]}"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": examples_b64[img_id]}})
        parts.append({"text": answers[img_id]})
    parts.append({"text": query_lead})
    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_query}})
    parts.append({"text": fmt})
    return parts


def create_batch_files():
    df = pd.read_excel(EXCEL_PATH)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    #if sample_size:
        #df_shuffled = df_shuffled.head(sample_size)

    total_images = len(df_shuffled)
    chunk_size = math.ceil(total_images / NUM_CHUNKS)
    all_payloads = []

    print(f"📦 Processing {total_images} images for {MODEL_TAG} | {LANG_TAG} | {STRATEGY_TAG}...")
    print("🖼️  Loading S6 example images...")
    examples_b64 = load_example_images()
    print()

    for index, row in df_shuffled.iterrows():
        raw_val = str(row.iloc[0]).strip()
        number_prefix = re.split(r'\.', raw_val)[0]
        context = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else "No context available."

        pattern = re.compile(rf"^{number_prefix}\..*", re.IGNORECASE)
        image_name = next((fn for fn in os.listdir(IMAGES_FOLDER) if pattern.match(fn)), None)

        if not image_name:
            print(f"⚠️  Image not found for ID: {number_prefix}, skipping.")
            continue

        img_path = os.path.join(IMAGES_FOLDER, image_name)
        b64_image = strip_exif_and_encode(img_path)

        if MODEL_TAG == "GPT":
            content = build_s6_content_openai(context, b64_image, examples_b64)
            payload = {
                "custom_id": f"id-{number_prefix}-{LANG_TAG}-{STRATEGY_TAG}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0,
                    "max_tokens": MAX_TOKENS
                }
            }
        elif MODEL_TAG == "Qwen":
            content = build_s6_content_openai(context, b64_image, examples_b64)
            payload = {
                "custom_id": f"id-{number_prefix}-{LANG_TAG}-{STRATEGY_TAG}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "Qwen/Qwen3.5-9B",
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0,
                    "max_tokens": MAX_TOKENS,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            }
        else:  # Gemini
            parts = build_s6_content_gemini(context, b64_image, examples_b64)
            payload = {
                "custom_id": f"id-{number_prefix}-{LANG_TAG}-{STRATEGY_TAG}",
                "request": {
                    "contents": [{"role": "user", "parts": parts}],
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