"""
Computes various metrics each model / strategy / language.

Ground truth:
  IDs  1–600  → Fake  → expected "Yes" (EN) or "Da" (RO)
  IDs 601–1200 → Real → expected "No"  (EN) or "Nu" (RO)
"""

import os
import re
import glob
import pandas as pd

RESULTS_FOLDER = r"D:\Bsc Thesis\Thesis_Results"

FAKE_IDS = range(1, 601)    # 1–600  inclusive
REAL_IDS = range(601, 1201) # 601–1200 inclusive

EXPECTED = {
    "EN": {"fake": "Yes", "real": "No"},
    "RO": {"fake": "Da",  "real": "Nu"},
}

VALID_FILE = re.compile(r'^(GPT|Gemini|Qwen)_(EN|RO)_(S[1-6])_clean\.xlsx$', re.IGNORECASE)


def get_file_meta(filename: str) -> tuple:
    m = re.match(r'^(GPT|Gemini|Qwen)_(EN|RO)_(S[1-6])_clean\.xlsx$', filename, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2).upper(), m.group(3).upper()
    return None, None, None


def get_decision_col(strategy: str) -> str:
    return "Verdict" if strategy == "S1" else "Image_Decision"


def expected_label(image_id: int, lang: str) -> str:
    if image_id in FAKE_IDS:
        return EXPECTED[lang]["fake"]
    if image_id in REAL_IDS:
        return EXPECTED[lang]["real"]
    return None


def compute_accuracy(path: str) -> dict | None:
    fname              = os.path.basename(path)
    model, lang, strat = get_file_meta(fname)
    if not model:
        return None

    df = pd.read_excel(path)

    decision_col = get_decision_col(strat)
    if decision_col not in df.columns:
        print(f" '{decision_col}' column missing in {fname}")
        return None

    if "Custom_ID" not in df.columns:
        print(f"'Custom_ID' column missing in {fname}")
        return None

    tp = 0  
    fp = 0  
    tn = 0  
    fn = 0  
    n_skipped = 0

    fake_label = EXPECTED[lang]["fake"]
    real_label = EXPECTED[lang]["real"]

    for _, row in df.iterrows():
        if str(row.get("REVIEW", "")).strip() == "UNPARSED":
            n_skipped += 1
            continue

        raw_id = row["Custom_ID"]
        try:
            image_id = int(raw_id)
        except (ValueError, TypeError):
            n_skipped += 1
            continue

        if image_id > 1200:
            continue

        answer = str(row.get(decision_col, "")).strip()
        if not answer or answer.lower() == "nan":
            n_skipped += 1
            continue

        exp = expected_label(image_id, lang)
        if exp is None:
            n_skipped += 1
            continue

        actually_fake   = (exp    == fake_label)
        predicted_fake  = (answer == fake_label)

        if predicted_fake and actually_fake:   tp += 1
        elif predicted_fake and not actually_fake: fp += 1
        elif not predicted_fake and not actually_fake: tn += 1
        else:                                  fn += 1

    n_total  = tp + fp + tn + fn
    accuracy  = round((tp + tn) / n_total * 100, 2)          if n_total > 0           else None
    precision = round(tp / (tp + fp) * 100, 2)               if (tp + fp) > 0         else None
    recall    = round(tp / (tp + fn) * 100, 2)               if (tp + fn) > 0         else None
    f1        = round(2 * tp / (2 * tp + fp + fn) * 100, 2)  if (2*tp + fp + fn) > 0  else None

    precision_real = round(tn / (tn + fn) * 100, 2)               if (tn + fn) > 0         else None
    recall_real    = round(tn / (tn + fp) * 100, 2)               if (tn + fp) > 0         else None
    f1_real        = round(2 * tn / (2 * tn + fn + fp) * 100, 2)  if (2*tn + fn + fp) > 0  else None

    macro_f1 = round((f1 + f1_real) / 2, 2) if (f1 is not None and f1_real is not None) else None

    input_tokens  = int(df["Prompt_Tokens"].sum())     if "Prompt_Tokens"     in df.columns else None
    output_tokens = int(df["Completion_Tokens"].sum()) if "Completion_Tokens" in df.columns else None
    total_tokens  = (input_tokens + output_tokens) if (input_tokens is not None and output_tokens is not None) else None

    prop_yes = prop_no = None
    if "Propaganda" in df.columns:
        parsed_df = df[df.get("REVIEW", pd.Series(dtype=str)).fillna("").astype(str).str.strip() != "UNPARSED"] \
                    if "REVIEW" in df.columns else df
        prop_vals = parsed_df["Propaganda"].dropna().astype(str).str.strip()
        prop_yes  = int((prop_vals.isin({"Yes", "Da"})).sum())
        prop_no   = int((prop_vals.isin({"No",  "Nu"})).sum())

    pref_left = pref_right = pref_neutral = None
    if "Political_Preference" in df.columns:
        parsed_df = df[df.get("REVIEW", pd.Series(dtype=str)).fillna("").astype(str).str.strip() != "UNPARSED"] \
                    if "REVIEW" in df.columns else df
        pref_vals    = parsed_df["Political_Preference"].dropna().astype(str).str.strip()
        pref_left    = int((pref_vals.isin({"Left",    "Stânga"})).sum())
        pref_right   = int((pref_vals.isin({"Right",   "Dreapta"})).sum())
        pref_neutral = int((pref_vals.isin({"Neutral", "Neutru"})).sum())

    return {
        "Model":          model,
        "Lang":           lang,
        "Strategy":       strat,
        "Accuracy":       accuracy,
        "Precision":      precision,       
        "Recall":         recall,          
        "F1":             f1,             
        "Precision_Real": precision_real,  
        "Recall_Real":    recall_real,     
        "F1_Real":        f1_real,         
        "Macro_F1":       macro_f1,        
        "TP":             tp,
        "FP":             fp,
        "TN":             tn,
        "FN":             fn,
        "Skipped":        n_skipped,
        "Total":          n_total,
        "Input_Tokens":   input_tokens,
        "Output_Tokens":  output_tokens,
        "Total_Tokens":   total_tokens,
        "Propaganda_Yes": prop_yes,
        "Propaganda_No":  prop_no,
        "Pref_Left":      pref_left,
        "Pref_Right":     pref_right,
        "Pref_Neutral":   pref_neutral,
    }


def run():
    files = sorted(glob.glob(os.path.join(RESULTS_FOLDER, "**", "*_clean.xlsx"), recursive=True))
    files = [f for f in files if VALID_FILE.match(os.path.basename(f))]

    print(f"Found {len(files)} file(s).\n")

    rows = []
    for f in files:
        result = compute_accuracy(f)
        if result:
            rows.append(result)
            tok_in  = f"{result['Input_Tokens']:,}"  if result['Input_Tokens']  is not None else "N/A"
            tok_out = f"{result['Output_Tokens']:,}" if result['Output_Tokens'] is not None else "N/A"
            print(f"  {result['Model']:8} {result['Lang']}  {result['Strategy']}  →  "
                  f"Acc: {str(result['Accuracy'])+'%':8}  "
                  f"P: {str(result['Precision'])+'%':8}  "
                  f"R: {str(result['Recall'])+'%':8}  "
                  f"F1: {str(result['F1'])+'%':8}  "
                  f"F1_Real: {str(result['F1_Real'])+'%':8}  "
                  f"Macro_F1: {str(result['Macro_F1'])+'%':8}  "
                  f"(n={result['Total']}, skipped={result['Skipped']})  "
                  f"[in={tok_in}, out={tok_out}]")

    if not rows:
        print("No results computed.")
        return

    df = pd.DataFrame(rows)

    for metric in ("Accuracy", "Precision", "Recall", "F1", "F1_Real", "Macro_F1"):
        print(f"\n{'=' * 60}")
        print(f"{metric.upper()} SUMMARY (%)")
        print("=" * 60)
        pivot = df.pivot_table(
            index=["Model", "Lang"],
            columns="Strategy",
            values=metric,
            aggfunc="first"
        )
        print(pivot.to_string())

    for col, label in (
        ("Propaganda_Yes", "PROPAGANDA YES COUNT"),
        ("Propaganda_No",  "PROPAGANDA NO COUNT"),
        ("Pref_Left",      "POLITICAL PREFERENCE — LEFT"),
        ("Pref_Right",     "POLITICAL PREFERENCE — RIGHT"),
        ("Pref_Neutral",   "POLITICAL PREFERENCE — NEUTRAL"),
    ):
        if col in df.columns and df[col].notna().any():
            print(f"\n{'=' * 60}")
            print(label)
            print("=" * 60)
            pivot = df.pivot_table(
                index=["Model", "Lang"],
                columns="Strategy",
                values=col,
                aggfunc="first"
            )
            print(pivot.to_string())

    if "Input_Tokens" in df.columns and df["Input_Tokens"].notna().any():
        for tok_col, label in (("Input_Tokens", "INPUT TOKENS"), ("Output_Tokens", "OUTPUT TOKENS"), ("Total_Tokens", "TOTAL TOKENS")):
            print(f"\n{'=' * 60}")
            print(f"{label} SUMMARY")
            print("=" * 60)
            pivot = df.pivot_table(
                index=["Model", "Lang"],
                columns="Strategy",
                values=tok_col,
                aggfunc="first"
            )
            print(pivot.to_string())

    out_path = os.path.join(RESULTS_FOLDER, "Metrics.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Per_File", index=False)

        for metric in ("Accuracy", "Precision", "Recall", "F1", "F1_Real", "Macro_F1"):
            pivot = df.pivot_table(
                index=["Model", "Lang"],
                columns="Strategy",
                values=metric,
                aggfunc="first"
            )
            pivot.to_excel(writer, sheet_name=metric)

        pivot_skipped = df.pivot_table(
            index=["Model", "Lang"],
            columns="Strategy",
            values="Skipped",
            aggfunc="first"
        )
        pivot_skipped.to_excel(writer, sheet_name="Skipped")

        for col, sheet in (
            ("Propaganda_Yes", "Propaganda_Yes"),
            ("Propaganda_No",  "Propaganda_No"),
        ):
            if col in df.columns and df[col].notna().any():
                pivot = df.pivot_table(
                    index=["Model", "Lang"],
                    columns="Strategy",
                    values=col,
                    aggfunc="first"
                )
                pivot.to_excel(writer, sheet_name=sheet)

        for col, sheet in (
            ("Pref_Left",    "Pref_Left"),
            ("Pref_Right",   "Pref_Right"),
            ("Pref_Neutral", "Pref_Neutral"),
        ):
            if col in df.columns and df[col].notna().any():
                pivot = df.pivot_table(
                    index=["Model", "Lang"],
                    columns="Strategy",
                    values=col,
                    aggfunc="first"
                )
                pivot.to_excel(writer, sheet_name=sheet)

        for tok_col, sheet in (
            ("Input_Tokens",  "Input_Tokens"),
            ("Output_Tokens", "Output_Tokens"),
            ("Total_Tokens",  "Total_Tokens"),
        ):
            if tok_col in df.columns and df[tok_col].notna().any():
                pivot = df.pivot_table(
                    index=["Model", "Lang"],
                    columns="Strategy",
                    values=tok_col,
                    aggfunc="first"
                )
                pivot.to_excel(writer, sheet_name=sheet)

    print(f"\n Saved: {out_path}")


if __name__ == "__main__":
    run()