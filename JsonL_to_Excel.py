import json
import os
import re
import pandas as pd
from glob import glob

MODEL_CONFIG = {
    "GPT": ("OpenAI",     "GPT"),
    "Qwen":   ("TogetherAI", "Qwen"),
    "Gemini": ("Gemini",     "Gemini"),
}

BASE_INPUT  = r"D:\BSc Thesis\batch_output"
BASE_OUTPUT = r"D:\BSc Thesis\Thesis_Results"

VALID_LANGUAGES  = ["EN", "RO"]
VALID_STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]


def ask_languages(prompt: str, valid: list[str]) -> list[str]:
    """Ask the user to pick one or more languages from a valid list."""
    valid_display = ", ".join(valid)
    while True:
        raw = input(f"{prompt}\n  Options: {valid_display} (or 'all')\n  > ").strip().upper()
        if raw == "ALL":
            return valid
        chosen = [v.strip() for v in raw.replace(",", " ").split()]
        invalid = [c for c in chosen if c not in valid]
        if invalid:
            print(f"  ⚠ Not recognised: {invalid}. Please try again.\n")
        elif not chosen:
            print("  ⚠ Please enter at least one option.\n")
        else:
            return chosen


def ask_strategies() -> list[str]:
    """Ask the user for strategy tags — accepts any S\w+ value, not just the predefined list."""
    print(f"Which strategy/strategies?")
    print(f"  Predefined: {', '.join(VALID_STRATEGIES)} (or 'all' for these)")
    print(f"  You can also type custom tags like S2_test, S2_v2, etc.")
    while True:
        raw = input("  > ").strip().upper()
        if raw == "ALL":
            return VALID_STRATEGIES
        chosen = [v.strip() for v in raw.replace(",", " ").split()]
        invalid = [c for c in chosen if not re.match(r'^S\w+$', c, re.IGNORECASE)]
        if invalid:
            print(f"  ⚠ Invalid strategy tags (must start with S): {invalid}. Please try again.\n")
        elif not chosen:
            print("  ⚠ Please enter at least one option.\n")
        else:
            return chosen


def clean(text):
    """Remove asterisks and insert space between letters and digits (e.g. 'Yes2' -> 'Yes 2')."""
    text = re.sub(r"\*", "", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    return text.strip()


def extract_fields(full_text: str) -> dict:
    result = {}

    # S1
    m = re.search(r"Verdict:\s*([^\n]+)", full_text, re.IGNORECASE)
    if m:
        result["Verdict"] = clean(m.group(1))

    # S2-S6
    # 1 – Image decision (EN + RO)
    m = re.search(r"(?:Image decision|Decizie imagine):\s*([^\n]+)", full_text, re.IGNORECASE)
    if m:
        result["Image_Decision"] = clean(m.group(1))

    # 2 – Detected cues (EN + RO)
    m = re.search(r"(?:Reasoning\s*[—\-]*\s*detected cues|Raționament\s*[—\-]*\s*indicii detectate):\s*([^\n]+)", full_text, re.IGNORECASE)
    if m:
        result["Detected_Cues"] = clean(m.group(1))

    # 3 – Political propaganda/manipulation (EN + RO)
    m = re.search(r"(?:Political propaganda[/\\]manipulation|Propagandă politică[/\\]manipulare):\s*([^\n]+)", full_text, re.IGNORECASE)
    if m:
        result["Propaganda"] = clean(m.group(1))

    # 4 – Manipulation technique (EN + RO)
    m = re.search(r"(?:Manipulation[/\\]Propaganda technique|Tehnică de manipulare[/\\]propagandă):\s*([^\n]+)", full_text, re.IGNORECASE)
    if m:
        result["Manipulation_Technique"] = clean(m.group(1))

    # 5 – Political preference (EN + RO)
    m = re.search(r"(?:Political preference|Preferință politică):\s*([^\n]+)", full_text, re.IGNORECASE)
    if m:
        result["Political_Preference"] = clean(m.group(1))

    # 1 (CoT) – Step-by-step reasoning (EN + RO) — S3/S5 
    m = re.search(r"(?:Step-by-step reasoning|Raționament pas cu pas):\s*([\s\S]+?)(?=\n\d+\.|\Z)", full_text, re.IGNORECASE)
    if m:
        result["CoT_Reasoning"] = clean(m.group(1))

    return result


def parse_jsonl(path: str, split_num: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                line = line.strip()
                if not line: continue

                data = json.loads(line)

                custom_id = data.get('custom_id') or data.get('key') or f"unknown-{line_num}"

                response_obj = data.get('response', {})
                response_body = response_obj.get('body', response_obj)

                full_text = "N/A"
                input_tokens = 0
                output_tokens = 0

                # GEMINI PATH
                if 'candidates' in response_body:
                    candidates = response_body.get('candidates', [])
                    if candidates:
                        full_text = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                    usage = response_body.get('usageMetadata', {})
                    input_tokens = usage.get('promptTokenCount', 0)
                    output_tokens = usage.get('candidatesTokenCount', usage.get('candidates_token_count', 0))

                # GPT / QWEN PATH
                elif 'choices' in response_body:
                    choices = response_body.get('choices', [])
                    if choices:
                        full_text = choices[0].get('message', {}).get('content', '')
                    usage = response_body.get('usage', {})
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)

                else:
                    full_text = str(data.get('error', 'Unknown Error or Empty Response'))

                row = {
                    "Custom_ID":         custom_id,
                    "Split":             f"split{split_num}",
                    "Full_Response":     full_text,
                    "Prompt_Tokens":     input_tokens,
                    "Completion_Tokens": output_tokens,
                    "Total_Tokens":      input_tokens + output_tokens
                }

                row.update(extract_fields(full_text))
                rows.append(row)

            except Exception as e:
                print(f"  Row {line_num} error: {e}")

    return rows


def process_model(model_name: str, languages: list[str], strategies: list[str]):

    provider, model_key = MODEL_CONFIG[model_name]
    current_input_folder = os.path.join(BASE_INPUT, provider)
    output_folder = os.path.join(BASE_OUTPUT, provider)

    is_gemini = (model_name.upper() == "GEMINI")

    if is_gemini:
        file_re = re.compile(
            rf"{re.escape(model_key)}_(?P<lang>[A-Z]{{2}})_(?P<strat>S[\w]+)_Split(?P<num>\d+)_output\.jsonl$",  # for Gemini S6 which uses multiple splits, as file size restriction is 500mbb
            re.IGNORECASE
        )
    else:
        file_re = re.compile(
            rf"{re.escape(model_key)}_(?P<lang>[A-Z]{{2}})_(?P<strat>S[\w]+)_Split(?P<num>\d+)_output\.jsonl$",
            re.IGNORECASE
        )

    groups = {}

    for lang in languages:
        for strat in strategies:
            strategy_folder = strat[:2] 
            strat_input_folder = os.path.join(current_input_folder, strategy_folder)

            if is_gemini:
                search_filename = f"{model_key}_{lang}_{strat}_Split*_output.jsonl" #for Gemini S6 which uses multiple splits, as file size restriction is 500mbb
            else:
                search_filename = f"{model_key}_{lang}_{strat}_Split*_output.jsonl"

            pattern = os.path.join(strat_input_folder, search_filename)
            print(f"Searching for: {pattern}")

            all_files = sorted(glob(pattern))

            if not all_files:
                print(f"No files found for {lang}_{strat}")
                continue

            print(f" Found {len(all_files)} file(s)")
            for fp in all_files:
                fname = os.path.basename(fp)
                m = file_re.search(fname)

                if not m:
                    print(f"File found but regex failed: {fname}")
                    continue

                split_no = m.group("num") or "1"
                try:
                    rows = parse_jsonl(fp, split_no)
                    groups.setdefault((lang, strat), []).extend(rows)
                    print(f" Processed {len(rows)} rows from {fname}")
                except Exception as e:
                    print(f" Failed to parse {fname}: {e}")


    os.makedirs(output_folder, exist_ok=True)
    for (lang, strat), rows in sorted(groups.items()):
        df = pd.DataFrame(rows)

        try:
            df['sort_id'] = df['Custom_ID'].str.extract(r'id-(\d+)').astype(int)
            df['Custom_ID'] = df['sort_id']
            df.sort_values("Custom_ID", ascending=True, inplace=True, ignore_index=True)
            df.drop(columns=['sort_id'], inplace=True)
        except Exception as e:
            print(f" Sorting failed for {lang}_{strat}: {e}")

        filename = f"{model_name}_{lang}_{strat}.xlsx"
        df.to_excel(os.path.join(output_folder, filename), index=False)
        print(f"SUCCESS: {filename} saved.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        model = sys.argv[1]
    else:
        model = input(f"Model name ({'/'.join(MODEL_CONFIG)}): ").strip()

    languages  = ask_languages("Which language(s)?", VALID_LANGUAGES)
    strategies = ask_strategies()

    print(f"\nRunning: {model} | Languages: {languages} | Strategies: {strategies}\n")
    process_model(model, languages, strategies)