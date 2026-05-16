"""
This python file generates various plots and analysis tables for the Results & Discussion section, as well as some of those included in the Appendix.
!!!! Metrics.py must be run before so all the metrics are in adequate folder.

Functions:
  1.  plot_accuracy_lines()         — line plots S1–S6 per model, EN / RO
  2.  plot_tpr_tnr()                — TPR vs TNR (fake-recall vs real-recall) bar chart
  3.  image_disagreement()          — top-N most disagreed-upon images across all models
  4.  consistency_heatmap()         — per-image correct/wrong heatmap for a given strategy
  5.  plot_technique_frequency()    — manipulation technique frequency by model
  6.  plot_cue_frequency()          — detected cue frequency: S2 vs S3/S5 (CoT)
  7.  plot_wordclouds()             — word cloud of CoT reasoning per model
  8.  language_asymmetric_refusals()— images where one language parsed, the other did not
  9.  en_ro_consistency()           — fraction of images with same verdict in EN and RO
  10. compute_kappa()               — Cohen's Kappa (model vs ground truth)

Dependencies:
  pip install matplotlib seaborn openpyxl pandas scikit-learn wordcloud
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

RESULTS_FOLDER = r"D:\Bsc Thesis\Thesis_Results"
PLOTS_FOLDER   = os.path.join(RESULTS_FOLDER, "Plots")
os.makedirs(PLOTS_FOLDER, exist_ok=True)

FAKE_IDS = set(range(1, 601))
REAL_IDS = set(range(601, 1201))

FAKE_LABELS = {"Yes", "Da"}

STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
MODELS     = ["GPT", "Gemini", "Qwen"]
LANGS      = ["EN", "RO"]

VALID_FILE = re.compile(r'^(GPT|Gemini|Qwen)_(EN|RO)_(S[1-6])_clean\.xlsx$', re.IGNORECASE)

CUE_LABELS = {
    "a": "Anatomical implausibility",
    "b": "Scene incoherence",
    "c": "Contextual implausibility",
    "d": "Digital artifacts",
    "e": "Textual errors in image",
}

TECHNIQUE_LABELS = {
    "1": "Emotional appeal",
    "2": "Name-calling / smear",
    "3": "False context / decontextualization",
    "4": "Appeal to authority",
    "5": "Simplification",
}

MODEL_COLORS  = {"GPT": "#1f77b4", "Gemini": "#ff7f0e", "Qwen": "#2ca02c"}
MODEL_MARKERS = {"GPT": "o",       "Gemini": "s",        "Qwen": "^"}

# font sizes
plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   15,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11,
    "figure.titlesize": 16,
})



#stopwords for word clouds
RO_STOPWORDS = {
    "și", "în", "că", "cu", "de", "la", "pe", "un", "o", "este", "sunt",
    "nu", "se", "sau", "din", "care", "mai", "pentru", "prin", "această",
    "acest", "acestea", "acestui", "acesta", "fi", "fost", "lui", "ei",
    "lor", "le", "îl", "îi", "cel", "cea", "ale", "al", "unor", "unui",
    "unei", "poate", "dacă", "când", "cum", "tot", "toate", "între",
    "față", "doar", "astfel", "deci", "dar", "iar", "totuși", "chiar",
    "foarte", "bine", "atât", "câte", "orice", "oricare", "fiecare",
    "imagine", "imaginea", "text", "context", "conținut", "răspuns",
    "the", "is", "are", "this", "that", "image", "shows", "there",
    "with", "has", "have", "not", "for", "from", "which", "been",
    "would", "could", "appears", "appear", "suggest", "suggests",
    "provided", "depict","suggesting","additionally","may","overall",
    "pare","sugerează", "prezintă","induce", "textul", "să fie",
    "mesajul","prezintă două", "indică", "ar", "par", "asemenea",
    "crea","loc"
}

#diacritice Ș ș ț Ț ă Ă î Î â Â
# load data
def load_all_files() -> dict:
    """
    Returns dict keyed by (model, lang, strategy) -> DataFrame.
    Skips UNPARSED rows and IDs > 1200 (used innitially with a different dataset that did not perform up to par).
    """
    files = sorted(glob.glob(os.path.join(RESULTS_FOLDER, "**", "*_clean.xlsx"), recursive=True))
    files = [f for f in files if VALID_FILE.match(os.path.basename(f))]

    if not files:
        print("❌ No *_clean.xlsx files found. Run Clean_Outputs.py first.")
        return {}

    data = {}
    for f in files:
        m = VALID_FILE.match(os.path.basename(f))
        model, lang, strat = m.group(1), m.group(2).upper(), m.group(3).upper()
        df = pd.read_excel(f)
        try:
            df["Custom_ID"] = df["Custom_ID"].astype(int)
            df = df[df["Custom_ID"] <= 1200].copy()
        except Exception:
            pass
        data[(model, lang, strat)] = df

    print(f"Loaded {len(data)} file(s).")
    return data

def get_decision_col(strat: str) -> str:
    return "Verdict" if strat == "S1" else "Image_Decision"

def get_expected(image_id: int, lang: str):
    if image_id in FAKE_IDS: return "Yes" if lang == "EN" else "Da"
    if image_id in REAL_IDS: return "No"  if lang == "EN" else "Nu"
    return None

def is_fake_prediction(answer: str) -> bool:
    return str(answer).strip() in FAKE_LABELS

def is_correct(answer: str, image_id: int, lang: str):
    exp = get_expected(image_id, lang)
    if exp is None or not answer or str(answer).strip().lower() == "nan":
        return None
    return str(answer).strip() == exp


# 0. LINE PLOTS => Macro F1 per strategy 
def plot_macro_f1_lines():
    """
    Two line charts (EN / RO): x = S1–S6, y = Macro F1 (%), one line per model.
    """
    metrics_path = os.path.join(RESULTS_FOLDER, "Metrics.xlsx")
    if not os.path.exists(metrics_path):
        print("Metrics.xlsx not found.")
        return
 
    df = pd.read_excel(metrics_path, sheet_name="Per_File")
    df = df[df["Strategy"].isin(STRATEGIES)]

    f1_col = next((c for c in df.columns if "macro" in c.lower() and "f1" in c.lower()), None)
    if f1_col is None:
        f1_col = next((c for c in df.columns if "f1" in c.lower()), None)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
 
    for ax, lang in zip(axes, ["EN", "RO"]):
        sub = df[df["Lang"] == lang]
        for model in MODELS:
            row = sub[sub["Model"].str.upper() == model.upper()].sort_values("Strategy")
            if row.empty:
                continue
            ax.plot(row["Strategy"], row[f1_col],
                    label=model,
                    color=MODEL_COLORS[model],
                    marker=MODEL_MARKERS[model],
                    linewidth=2, markersize=7)
        ax.axhline(50, color="red", linestyle=":", linewidth=1.5, label="Random chance (50%)")
        ax.set_title(f"Macro F1 — {lang}", fontsize=18)
        ax.set_xlabel("Strategy", fontsize=16)
        ax.set_ylabel("Macro F1 (%)", fontsize=16)
        ax.set_ylim(40, 100)
        ax.legend(fontsize=13)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
 
    plt.suptitle("Deepfake Detection Macro F1 by Strategy and Language (Primary Metric)", fontsize=18)
    plt.tight_layout()
    out = os.path.join(PLOTS_FOLDER, "macro_f1_lines.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {out}")

# 1. LINE PLOTS => Accuracy F1 per strategy 
def plot_accuracy_lines():
    """
    Two line charts (EN / RO): x = S1–S6, y = Accuracy (%), one line per model.
    """
    metrics_path = os.path.join(RESULTS_FOLDER, "Metrics.xlsx")
    if not os.path.exists(metrics_path):
        print("Metrics.xlsx not found.")
        return
 
    df = pd.read_excel(metrics_path, sheet_name="Per_File")
    df = df[df["Strategy"].isin(STRATEGIES)]
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
 
    for ax, lang in zip(axes, ["EN", "RO"]):
        sub = df[df["Lang"] == lang]
        for model in MODELS:
            row = sub[sub["Model"].str.upper() == model.upper()].sort_values("Strategy")
            if row.empty:
                continue
            ax.plot(row["Strategy"], row["Accuracy"],
                    label=model,
                    color=MODEL_COLORS[model],
                    marker=MODEL_MARKERS[model],
                    linewidth=2, markersize=7)
        ax.axhline(50, color="red", linestyle=":", linewidth=1.5, label="Random chance (50%)")
        ax.set_title(f"Accuracy — {lang}", fontsize=18)
        ax.set_xlabel("Strategy", fontsize=16)
        ax.set_ylabel("Accuracy (%)", fontsize=16)
        ax.set_ylim(40, 100)
        ax.legend(fontsize=13)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
 
    plt.suptitle("Deepfake Detection Accuracy by Strategy and Language", fontsize=18)
    plt.tight_layout()
    out = os.path.join(PLOTS_FOLDER, "accuracy_lines.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
 
# 2. TPR / TNR => Fake vs Real recall split
def plot_tpr_tnr(data: dict):
    """
    Two line charts (EN / RO): x = S1–S6, y = Sensitivity (%).
    Sensitivity = TPR = % of fake images correctly detected.
    Specificity is almost perfect (~100%) so is not plotted separately.
    """
    rows = []
    for (model, lang, strat), df in data.items():
        dec_col = get_decision_col(strat)
        if dec_col not in df.columns:
            continue
        tp = fn = 0
        for _, row in df.iterrows():
            if str(row.get("REVIEW", "")).strip() == "UNPARSED":
                continue
            iid = row.get("Custom_ID")
            try:
                iid = int(iid)
            except (ValueError, TypeError):
                continue
            ans = str(row.get(dec_col, "")).strip()
            if not ans or ans.lower() == "nan":
                continue
            actually_fake  = iid in FAKE_IDS
            predicted_fake = is_fake_prediction(ans)
            if predicted_fake and actually_fake:  tp += 1
            elif not predicted_fake and actually_fake: fn += 1
 
        sensitivity = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else None
        rows.append({"Model": model, "Lang": lang, "Strategy": strat,
                     "Sensitivity": sensitivity})
 
    df_sens = pd.DataFrame(rows).dropna(subset=["Sensitivity"])
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
 
    for ax, lang in zip(axes, ["EN", "RO"]):
        sub = df_sens[df_sens["Lang"] == lang]
        for model in MODELS:
            row = sub[sub["Model"].str.upper() == model.upper()].sort_values("Strategy")
            if row.empty:
                continue
            ax.plot(row["Strategy"], row["Sensitivity"],
                    label=model,
                    color=MODEL_COLORS[model],
                    marker=MODEL_MARKERS[model],
                    linewidth=2, markersize=7)
        ax.axhline(50, color="red", linestyle=":", linewidth=1.5, label="Random chance (50%)")
        ax.set_title(f"Sensitivity — {lang}", fontsize=18)
        ax.set_xlabel("Strategy", fontsize=16)
        ax.set_ylabel("Sensitivity (%)", fontsize=16)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=13)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
 
    plt.suptitle("Fake Image Detection Sensitivity by Strategy and Language", fontsize=18)
    plt.tight_layout()
    out = os.path.join(PLOTS_FOLDER, "sensitivity_lines.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {out}")

# 3. IMAGE DISAGREEMENT ANALYSIS
def image_disagreement(data: dict, top_n: int = 15) -> pd.DataFrame:
    """
    Disagreement score per image across all (model, lang, strategy) runs.
    Score = 1.0 if the vote is perfectly split (half fake / half real), 0.0 if unanimous.
    Saves a bar chart and a CSV of the top_n most disagreed images.
    """
    votes = {}
    for (model, lang, strat), df in data.items():
        dec_col = get_decision_col(strat)
        if dec_col not in df.columns:
            continue
        for _, row in df.iterrows():
            if str(row.get("REVIEW", "")).strip() == "UNPARSED":
                continue
            iid = row.get("Custom_ID")
            try:
                iid = int(iid)
            except (ValueError, TypeError):
                continue
            ans = str(row.get(dec_col, "")).strip()
            if not ans or ans.lower() == "nan":
                continue
            if iid not in votes:
                votes[iid] = {"fake": 0, "real": 0}
            if is_fake_prediction(ans):
                votes[iid]["fake"] += 1
            else:
                votes[iid]["real"] += 1
 
    rows = []
    for iid, v in votes.items():
        total = v["fake"] + v["real"]
        if total == 0:
            continue
        fake_frac    = v["fake"] / total
        disagreement = 1 - abs(fake_frac - 0.5) * 2  # 1.0 = perfectly split
        true_label   = "Fake" if iid in FAKE_IDS else "Real"
        rows.append({
            "Image_ID":      iid,
            "True_Label":    true_label,
            "Fake_Votes":    v["fake"],
            "Real_Votes":    v["real"],
            "Total_Votes":   total,
            "Fake_Pct":      round(fake_frac * 100, 1),
            "Disagreement":  round(disagreement, 3),
        })
 
    df_dis = (pd.DataFrame(rows)
                .sort_values("Disagreement", ascending=False)
                .reset_index(drop=True))
    top = df_dis.head(top_n)
 
    # distribution plot: disagreement score by true label
    fake_scores = df_dis[df_dis["True_Label"] == "Fake"]["Disagreement"]
    real_scores = df_dis[df_dis["True_Label"] == "Real"]["Disagreement"]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(real_scores, bins=bins, alpha=0.6, color="#2ecc71", label="True label: Real",
            edgecolor="white", linewidth=0.5)
    ax.hist(fake_scores, bins=bins, alpha=0.6, color="#e74c3c", label="True label: Fake",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Disagreement Score (0 = unanimous, 1 = perfectly split)", fontsize=18)
    ax.set_ylabel("Number of images", fontsize=18)
    ax.set_title("Distribution of Per-Image Disagreement Scores by True Label\n"
                 "(aggregated across all models, strategies, and languages)", fontsize=18)
    ax.legend(fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
 
    # summary stats 
    ax.axvline(fake_scores.mean(), color="#c0392b", linestyle="--", linewidth=1.2,
               label=f"Fake mean ({fake_scores.mean():.2f})")
    ax.axvline(real_scores.mean(), color="#27ae60", linestyle="--", linewidth=1.2,
               label=f"Real mean ({real_scores.mean():.2f})")
    ax.legend(fontsize=15)
    plt.tight_layout()
    out_png = os.path.join(PLOTS_FOLDER, "image_disagreement.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {out_png}")
    print(f"   Fake mean disagreement: {fake_scores.mean():.3f} | Real mean: {real_scores.mean():.3f}")
    print(f"   Perfectly split (score=1.0): {(df_dis['Disagreement']==1.0).sum()} images "
          f"({df_dis[df_dis['Disagreement']==1.0]['True_Label'].value_counts().to_dict()})")
    print(f"   Score > 0.8: {(df_dis['Disagreement']>0.8).sum()} images "
          f"({df_dis[df_dis['Disagreement']>0.8]['True_Label'].value_counts().to_dict()})")
    print(f"   Score < 0.2 (near-consensus): {(df_dis['Disagreement']<0.2).sum()} images "
          f"({df_dis[df_dis['Disagreement']<0.2]['True_Label'].value_counts().to_dict()})")
 
    out_csv = os.path.join(PLOTS_FOLDER, "image_disagreement.csv")
    df_dis.to_csv(out_csv, index=False)
    print(f" Saved: {out_csv}")
 
    return top
 
 

# 4. PER-IMAGE CONSISTENCY HEATMAP
def consistency_heatmap(data: dict, strategy: str = "S2", top_n_images: int = 60):
    """
    Heatmap: rows = image IDs, columns = model × language (6 combos).
    Green = correct, Red = wrong, Grey = missing.
    Images sorted by disagreement (most contested first).
    """
    keys    = [(m, l) for m in MODELS for l in LANGS]
    records = {k: {} for k in keys}
 
    for (model, lang, strat), df in data.items():
        if strat != strategy:
            continue
        dec_col = get_decision_col(strat)
        if dec_col not in df.columns:
            continue
        for _, row in df.iterrows():
            if str(row.get("REVIEW", "")).strip() == "UNPARSED":
                continue
            iid = row.get("Custom_ID")
            try:
                iid = int(iid)
            except (ValueError, TypeError):
                continue
            ans     = str(row.get(dec_col, "")).strip()
            correct = is_correct(ans, iid, lang)
            records[(model, lang)][iid] = correct
 
    all_ids = sorted(set(iid for r in records.values() for iid in r))
 
    def disagreement_score(iid):
        votes = sum(1 for r in records.values() if r.get(iid) is True)
        total = sum(1 for r in records.values() if r.get(iid) is not None)
        if total == 0:
            return 0
        frac = votes / total
        return 1 - abs(frac - 0.5) * 2
 
    all_ids = sorted(all_ids, key=disagreement_score, reverse=True)[:top_n_images]
 
    col_names = [f"{m}\n{l}" for m, l in keys]
    matrix    = pd.DataFrame(index=all_ids, columns=col_names, dtype=float)
 
    for (model, lang), id_map in records.items():
        col = f"{model}\n{lang}"
        for iid in all_ids:
            v = id_map.get(iid, None)
            matrix.loc[iid, col] = 1.0 if v is True else (0.0 if v is False else np.nan)
 
    fig_h = max(10, top_n_images // 4)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    sns.heatmap(
        matrix, ax=ax,
        cmap=["#e74c3c", "#2ecc71"],
        linewidths=0.2, linecolor="#eeeeee",
        cbar_kws={"ticks": [0, 1], "label": "0 = Wrong   1 = Correct"},
        yticklabels=all_ids,
    )
    ax.set_title(f"Per-Image Correctness — Strategy {strategy} (sorted by disagreement)", fontsize=12)
    ax.set_xlabel("Model × Language")
    ax.set_ylabel("Image ID")
    plt.tight_layout()
    out = os.path.join(PLOTS_FOLDER, f"consistency_heatmap_{strategy}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
 

# 5. MANIPULATION TECHNIQUE FREQUENCY
def plot_technique_frequency(data: dict):
    """
    Bar chart: % of propaganda-flagged responses tagged with each technique (1–5),
    aggregated across all models, strategies, and languages.
    Bars can sum to >100% because multiple techniques can be assigned per image.
    """
    PROPAGANDA_YES = {"Yes", "Da"}
    counts      = Counter()
    prop_total  = 0
 
    for (model, lang, strat), df in data.items():
        if "Manipulation_Technique" not in df.columns:
            continue
        if "Propaganda" not in df.columns:
            continue
        filtered = df[df["Propaganda"].isin(PROPAGANDA_YES)]
        prop_total += len(filtered)
        for val in filtered["Manipulation_Technique"].dropna():
            for num in re.findall(r"[1-5]", str(val)):
                counts[num] += 1
 
    techniques = ["1", "2", "3", "4", "5"]
    vals = [counts.get(t, 0) / prop_total * 100 if prop_total > 0 else 0
            for t in techniques]
    labels = [f"{t}. {TECHNIQUE_LABELS[t]}" for t in techniques]
 
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, vals, color="#4C72B0", alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
 
    ax.set_ylabel("% of propaganda-flagged responses (all models, strategies & languages)", fontsize=16)
    ax.set_title("Manipulation Technique Frequency\n"
                 "(% of propaganda-flagged images; multiple techniques possible per image)", fontsize=16)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=13)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(PLOTS_FOLDER, "technique_frequency.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {out}")
 
 

# 6. CUE FREQUENCY ANALYSIS — S2 vs S3/S5 (CoT)
def plot_cue_frequency(data: dict):
    """
    Side-by-side bar chart: cue frequency (a–e) for S2 vs S3/S5 combined.
    Directly measures whether the explicit diacritic hint in CoT strategies
    increases detection of cue 'e' (Textual errors).
    """
    s2_counts   = Counter()
    s4_counts   = Counter()
    cot_counts  = Counter()
    s2_fake_total  = 0
    s4_fake_total  = 0
    cot_fake_total = 0
 
    # cue 'e' breakdown by language (all strategies combined)
    cue_e_by_lang   = {"EN": 0, "RO": 0}
    total_by_lang   = {"EN": 0, "RO": 0}
 
    for (model, lang, strat), df in data.items():
        if "Detected_Cues" not in df.columns:
            continue
        # ise rows where cues were actually detected 
        fake_df = df[~df["Detected_Cues"].astype(str).str.strip().isin(["N", "n", ""])]
 
        # cue 'e' by language — all strategies
        total_by_lang[lang] += len(fake_df)
        for val in fake_df["Detected_Cues"].dropna():
            if "e" in re.findall(r"[a-eA-E]", str(val).lower()):
                cue_e_by_lang[lang] += 1
 
        if strat == "S2":
            s2_fake_total += len(fake_df)
            for val in fake_df["Detected_Cues"].dropna():
                s2_counts.update(l.lower() for l in re.findall(r"[a-eA-E]", str(val)))
        elif strat == "S4":
            s4_fake_total += len(fake_df)
            for val in fake_df["Detected_Cues"].dropna():
                s4_counts.update(l.lower() for l in re.findall(r"[a-eA-E]", str(val)))
        elif strat in ("S3", "S5"):
            cot_fake_total += len(fake_df)
            for val in fake_df["Detected_Cues"].dropna():
                cot_counts.update(l.lower() for l in re.findall(r"[a-eA-E]", str(val)))
 
    print("\n── Cue 'e' (Textual errors) by language (all strategies, fake-flagged only) ──")
    for lang in ("EN", "RO"):
        total = total_by_lang[lang]
        count = cue_e_by_lang[lang]
        pct   = count / total * 100 if total > 0 else 0
        print(f"  {lang}: {count} / {total} fake-flagged responses → {pct:.1f}%")
 
    cues   = ["a", "b", "c", "d", "e"]
    x      = np.arange(len(cues))
    width  = 0.25
    labels = [f"{c}. {CUE_LABELS[c]}" for c in cues]
 
    s2_vals  = [s2_counts.get(c, 0)  / s2_fake_total  * 100 if s2_fake_total  > 0 else 0 for c in cues]
    s4_vals  = [s4_counts.get(c, 0)  / s4_fake_total  * 100 if s4_fake_total  > 0 else 0 for c in cues]
    cot_vals = [cot_counts.get(c, 0) / cot_fake_total * 100 if cot_fake_total > 0 else 0 for c in cues]
 
    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - width, s2_vals,  width, label="S2 (bare label)",    color="#3498db", alpha=0.85)
    b2 = ax.bar(x,         s4_vals,  width, label="S4 (Persona)",       color="#2ecc71", alpha=0.85)
    b3 = ax.bar(x + width, cot_vals, width, label="S3 + S5 (CoT hint)", color="#e67e22", alpha=0.85)
 
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=12)
 
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=13)
    ax.set_ylabel("% of fake-flagged responses (all models × languages)", fontsize=16)
    ax.set_title("Detected Cue Frequency: S2 vs S4 (Persona) vs S3/S5 (CoT)\n"
                 "(% of fake-verdict images; multiple cues possible per image)", fontsize=16)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.legend(fontsize=13)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(PLOTS_FOLDER, "cue_frequency.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
 
 
# 7. WORD CLOUD — CoT Reasoning - in the Appendix
def plot_wordclouds(data: dict):
    """
    Two word clouds: one combining all EN CoT reasoning across all models,
    one combining all RO CoT reasoning across all models.
    """
    en_stopwords = STOPWORDS | RO_STOPWORDS
    ro_stopwords = STOPWORDS | RO_STOPWORDS
 
    texts = {"EN": [], "RO": []}
    for (model, lang, strat), df in data.items():
        if strat not in ("S3", "S5"):
            continue
        if "CoT_Reasoning" not in df.columns:
            continue
        dec_col = get_decision_col(strat)
        if dec_col not in df.columns:
            continue
    
        filtered = df[df[dec_col].isin(FAKE_LABELS)]
        texts[lang].extend(filtered["CoT_Reasoning"].dropna().astype(str).tolist())
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    configs = [
        ("EN", "English Reasoning (S3 + S5, all models)", en_stopwords, "viridis"),
        ("RO", "Romanian Reasoning (S3 + S5, all models)", ro_stopwords, "plasma"),
    ]
 
    for ax, (lang, title, sw, cmap) in zip(axes, configs):
        combined = " ".join(texts[lang])
        if not combined.strip():
            ax.set_title(f"{title} — no data")
            ax.axis("off")
            continue
        wc = WordCloud(
            width=800, height=500,
            background_color="white",
            stopwords=sw,
            max_words=150,
            colormap=cmap,
        ).generate(combined)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
 
    plt.suptitle("Word Clouds: Step-by-step CoT Reasoning by Language", fontsize=14)
    plt.tight_layout()
    out = os.path.join(PLOTS_FOLDER, "wordclouds_cot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {out}")
 

# 8. LANGUAGE-ASYMMETRIC REFUSALS
def language_asymmetric_refusals() -> pd.DataFrame:
    """
    Finds images where one language produced a valid response but the other
    was UNPARSED / refused — for the exact same model, strategy, and image.
    """
    raw = load_raw_files()
 
    rows = []
    for model in MODELS:
        for strat in STRATEGIES:
            en_df = raw.get((model, "EN", strat))
            ro_df = raw.get((model, "RO", strat))
            if en_df is None or ro_df is None:
                continue
 
            en_idx = en_df.set_index("Custom_ID")
            ro_idx = ro_df.set_index("Custom_ID")
            common = set(en_idx.index) & set(ro_idx.index)
 
            for iid in common:
                en_up = str(en_idx.loc[iid].get("REVIEW", "")).strip() == "UNPARSED"
                ro_up = str(ro_idx.loc[iid].get("REVIEW", "")).strip() == "UNPARSED"
                if en_up == ro_up:
                    continue  # both parsed or both unparsed — not interesting
                rows.append({
                    "Model":       model,
                    "Strategy":    strat,
                    "Image_ID":    iid,
                    "True_Label":  "Fake" if iid in FAKE_IDS else "Real",
                    "EN_Status":   "UNPARSED" if en_up else "OK",
                    "RO_Status":   "UNPARSED" if ro_up else "OK",
                    "Direction":   "EN fails, RO ok" if en_up else "RO fails, EN ok",
                })
 
    df_asym = pd.DataFrame(rows)
 
    if df_asym.empty:
        print("No language-asymmetric refusals found.")
        return df_asym
 
    print(f"\nLanguage-asymmetric refusals: {len(df_asym)} cases")
    print(df_asym.groupby(["Model", "Direction"]).size().rename("Count").to_string())
 
    out_csv = os.path.join(PLOTS_FOLDER, "asymmetric_refusals.csv")
    df_asym.to_csv(out_csv, index=False)
    print(f" Saved: {out_csv}")
 
    summary = df_asym.groupby(["Model", "Direction"]).size().reset_index(name="Count")
    directions = ["EN fails, RO ok", "RO fails, EN ok"]
    x     = np.arange(len(MODELS))
    width = 0.35
    dir_colors = {"EN fails, RO ok": "#e74c3c", "RO fails, EN ok": "#3498db"}
 
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, direction in enumerate(directions):
        vals = []
        for model in MODELS:
            row = summary[(summary["Model"] == model) & (summary["Direction"] == direction)]
            vals.append(int(row["Count"].values[0]) if not row.empty else 0)
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=direction, color=dir_colors[direction], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        str(v), ha="center", va="bottom", fontsize=16)
 
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=18)
    ax.set_ylabel("Number of asymmetric refusal cases", fontsize=18)
    ax.set_title("Language-Asymmetric Refusals by Model and Direction\n"
                 "(cases where one language was parsed but the other was UNPARSED)", fontsize=18)
    ax.legend(fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_png = os.path.join(PLOTS_FOLDER, "asymmetric_refusals.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {out_png}")
 
    return df_asym
 

# 9. EN <=> RO CONSISTENCY
def en_ro_consistency(data: dict) -> pd.DataFrame:
    """
    For each (model, strategy): fraction of images where the EN and RO runs
    gave the same verdict (both fake or both real), regardless of correctness.
    Low score = the model's judgment is language-driven rather than image-driven.
    """
    rows = []
    for model in MODELS:
        for strat in STRATEGIES:
            en_df = data.get((model, "EN", strat))
            ro_df = data.get((model, "RO", strat))
            if en_df is None or ro_df is None:
                continue
            dec_col = get_decision_col(strat)
            if dec_col not in en_df.columns or dec_col not in ro_df.columns:
                continue

            en_map = en_df.set_index("Custom_ID")[dec_col].to_dict()
            ro_map = ro_df.set_index("Custom_ID")[dec_col].to_dict()

            agree = total = 0
            for iid in set(en_map) & set(ro_map):
                en_ans = str(en_map[iid]).strip()
                ro_ans = str(ro_map[iid]).strip()
                if not en_ans or en_ans.lower() == "nan":
                    continue
                if not ro_ans or ro_ans.lower() == "nan":
                    continue
                total += 1
                if is_fake_prediction(en_ans) == is_fake_prediction(ro_ans):
                    agree += 1

            consistency = round(agree / total * 100, 2) if total > 0 else None
            rows.append({"Model": model, "Strategy": strat,
                         "Consistency_%": consistency, "N": total})

    df_con = pd.DataFrame(rows)
    pivot  = df_con.pivot_table(
        index="Model", columns="Strategy",
        values="Consistency_%", aggfunc="first"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", annot_kws={"size": 14},
                cmap="RdYlGn", vmin=50, vmax=100,
                linewidths=0.5,
                cbar_kws={"label": "Consistency (%)", "shrink": 0.8})
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel("Strategy", fontsize=16)
    ax.set_ylabel("Model", fontsize=16)
    ax.set_title("EN ↔ RO Decision Consistency (%) — same image, same verdict across languages?", fontsize=16)
    plt.tight_layout()
    out = os.path.join(PLOTS_FOLDER, "en_ro_consistency.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {out}")
    print("\nEN ↔ RO Consistency pivot:")
    print(pivot.to_string())
    return df_con



# 10. COHEN'S KAPPA
def compute_kappa(data: dict) -> pd.DataFrame:
    """
    Cohen's Kappa between model predictions and ground truth per (model, lang, strategy).
    Kappa adjusts for chance, making it a more rigorous complement to accuracy.
    """
    rows = []
    for (model, lang, strat), df in data.items():
        dec_col    = get_decision_col(strat)
        if dec_col not in df.columns:
            continue
        fake_label = "Yes" if lang == "EN" else "Da"

        y_true, y_pred = [], []
        for _, row in df.iterrows():
            if str(row.get("REVIEW", "")).strip() == "UNPARSED":
                continue
            iid = row.get("Custom_ID")
            try:
                iid = int(iid)
            except (ValueError, TypeError):
                continue
            ans = str(row.get(dec_col, "")).strip()
            if not ans or ans.lower() == "nan":
                continue
            exp = get_expected(iid, lang)
            if exp is None:
                continue
            y_true.append(1 if exp == fake_label else 0)
            y_pred.append(1 if is_fake_prediction(ans) else 0)

        if len(set(y_true)) < 2 or len(set(y_pred)) < 2 or len(y_true) == 0:
            kappa = None
        else:
            kappa = round(cohen_kappa_score(y_true, y_pred), 4)
        rows.append({"Model": model, "Lang": lang, "Strategy": strat, "Kappa": kappa})

    df_kappa = pd.DataFrame(rows)
    pivot    = df_kappa.pivot_table(
        index=["Model", "Lang"], columns="Strategy",
        values="Kappa", aggfunc="first"
    )

    print("\nCOHEN'S KAPPA (model vs ground truth):")
    print(pivot.to_string())

    out = os.path.join(r"D:\BSc Thesis\Thesis_Results\Plots", "Kappa.xlsx")
    pivot.to_excel(out)
    print(f" Saved: {out}")
    return df_kappa


# 11. POLITICAL PREFERENCE FREQUENCY
def plot_political_preference(data: dict):
    """
    Grouped bar chart: percentage of responses assigned each political preference
    (Left / Right / Neutral / None) per model × language, aggregated across all strategies.
    Y-axis is percentage of total responses for that (model, lang) pair.
    """
    RO_MAP = {"stânga": "Left", "dreapta": "Right", "neutru": "Neutral"}
 
    counts = {m: {l: Counter() for l in LANGS} for m in MODELS}
    totals = {m: {l: 0 for l in LANGS} for m in MODELS}
 
    for (model, lang, strat), df in data.items():
        if "Political_Preference" not in df.columns:
            continue
        for val in df["Political_Preference"]:
            totals[model][lang] += 1
            if pd.isna(val) or str(val).strip() == "":
                counts[model][lang]["No Preference"] += 1
                continue
            v = str(val).strip().lower()
            if lang == "RO":
                v = RO_MAP.get(v, v.capitalize())
            else:
                v = v.capitalize()
            if v in ("Left", "Right"):
                counts[model][lang][v] += 1
            else:
                counts[model][lang]["No Preference"] += 1
 
    categories = ["Left", "Right", "No Preference"]
    x          = np.arange(len(categories))
    width      = 0.13
    fig, ax    = plt.subplots(figsize=(14, 5))
 
    offsets = {"GPT EN": -2.5, "GPT RO": -1.5, "Gemini EN": -0.5,
               "Gemini RO": 0.5, "Qwen EN": 1.5, "Qwen RO": 2.5}
    colors  = {"GPT EN": "#1f77b4", "GPT RO": "#aec7e8",
               "Gemini EN": "#ff7f0e", "Gemini RO": "#ffbb78",
               "Qwen EN": "#2ca02c", "Qwen RO": "#98df8a"}
 
    for model in MODELS:
        for lang in LANGS:
            key   = f"{model} {lang}"
            total = totals[model][lang] or 1  
            pcts  = [counts[model][lang].get(c, 0) / total * 100 for c in categories]
            bars  = ax.bar(x + offsets[key] * width, pcts, width,
                           label=key, color=colors[key], alpha=0.85)
            for bar, pct in zip(bars, pcts):
                if pct >= 1.0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.4,
                            f"{pct:.1f}%", ha="center", va="bottom", fontsize=11)
 
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=15)
    ax.set_ylabel("% of total responses (all strategies)", fontsize=18)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_title("Political Leaning Assigned by Model and Language (% of responses)", fontsize=18)
    ax.legend(fontsize=13, ncol=3)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(PLOTS_FOLDER, "political_preference.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {out}")


def run_all():
    print("=" * 60)
    print("Loading clean Excel files...")
    data = load_all_files()
    if not data:
        return

    print("\n[1/12] Accuracy line plots...")
    plot_accuracy_lines()

    print("\n[2/12] TPR / TNR split...")
    plot_tpr_tnr(data)

    print("\n[3/12] Image disagreement analysis...")
    image_disagreement(data)

    print("\n[4/12] Consistency heatmap (S2)...")
    consistency_heatmap(data, strategy="S2")

    print("\n[5/12] Manipulation technique frequency...")
    plot_technique_frequency(data)

    print("\n[6/12] Cue frequency (S2 vs S3/S5)...")
    plot_cue_frequency(data)

    print("\n[7/12] Word clouds (CoT reasoning)...")
    plot_wordclouds(data)

    print("\n[8/12] Language-asymmetric refusals...")
    language_asymmetric_refusals()

    print("\n[9/12] EN ↔ RO consistency...")
    en_ro_consistency(data)

    print("\n[10/12] Cohen's Kappa...")
    compute_kappa(data)

    print("\n[11/12] Political preference frequency...")
    plot_political_preference(data)

    DATASET_EXCEL = os.path.join(r"D:\BSc Thesis\Data", "Thesis Photos Descriptions.xlsx")
    print("\n[12/12] Empty context spot-check...")

    print(f"\n{'=' * 60}")
    print(f" Done.")


if __name__ == "__main__":
    import sys

    # run all analysis: python Analysis.py
    # run specific analysis: python Analysis.py 1 3 7

    DATASET_EXCEL = os.path.join(r"D:\BSc Thesis\Data", "Thesis Photos Descriptions.xlsx")

    selected = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(range(1, 13))

    print("=" * 60)
    print(f"Running analyses: {selected}")
    data = load_all_files()
    print("\nFiles loaded:")
    files = sorted(glob.glob(os.path.join(RESULTS_FOLDER, "**", "*_clean.xlsx"), recursive=True))
    files = [f for f in files if VALID_FILE.match(os.path.basename(f))]
    for f in files:
        print(f"  {f}")
    print(f"Total: {len(files)} files\n")
    if not data:
        raise SystemExit

    steps = {
        0:  lambda: plot_macro_f1_lines(),
        1:  lambda: plot_accuracy_lines(),
        2:  lambda: plot_tpr_tnr(data),
        3:  lambda: image_disagreement(data),
        4:  lambda: consistency_heatmap(data, strategy="S2"),
        5:  lambda: plot_technique_frequency(data),
        6:  lambda: plot_cue_frequency(data),
        7:  lambda: plot_wordclouds(data),
        8:  lambda: language_asymmetric_refusals(),
        9:  lambda: en_ro_consistency(data),
        10: lambda: compute_kappa(data),
        11: lambda: plot_political_preference(data),
    }

    for n in selected:
        print(f"\n[{n}] Running...")
        steps[n]()

    print(f"\n{'=' * 60}")
    print(f"Done.")
