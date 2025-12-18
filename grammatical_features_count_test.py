import torch
import os
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

# Set Up
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Models, tokenizer, and create custom BOS for Nahuatl
v_model = MBartForConditionalGeneration.from_pretrained(os.path.join(os.getcwd(), "Vanilla")).to(device)
tt_model = MBartForConditionalGeneration.from_pretrained(os.path.join(os.getcwd(), "TransTokenized")).to(device)
tokenizer = MBart50TokenizerFast.from_pretrained(os.path.join(os.getcwd(), "Vanilla"))

tokenizer.src_lang = "es_XX"
if "nah_XX" not in tokenizer.lang_code_to_id:
    tokenizer.lang_code_to_id["nah_XX"] = len(tokenizer.lang_code_to_id)

# Load Dataset
dataset = load_dataset("Thermostatic/Axolotl_Classical_Nahuatl_INALI")

# Read CSV
predictions_df = pd.read_csv(predictions_df = pd.read_csv(os.path.join(os.getcwd(), "validation_predictions.csv")))

tt_preds = [str(x) if pd.notna(x) else "" for x in predictions_df["Trans-Tokenization Prediction"].tolist()]
v_preds = [str(x) if pd.notna(x) else "" for x in predictions_df["Vanilla Prediction"].tolist()]
references = [str(x) if pd.notna(x) else "" for x in predictions_df["Reference"].tolist()]


# Grammar Features
def count_affix(t_preds, v_preds, refs, affix):
    counts_t_pred = {a:0 for a in affix}
    counts_v_pred = {a:0 for a in affix}
    counts_ref = {a:0 for a in affix}

    for t_pred, v_pred, ref in zip(t_preds, v_preds, refs):
        for a in affix:
            pattern = re.compile(r"\b" + re.escape(a))
            counts_t_pred[a] += len(pattern.findall(t_pred))
            counts_v_pred[a] += len(pattern.findall(v_pred))
            counts_ref[a] += len(pattern.findall(ref))
    return counts_t_pred, counts_v_pred, counts_ref


pronouns = ["ne", "te"]
adverbs = ["pan", "yakapan", "ko", "wan", "kan", "yan", "ixko"]
prepositions = ["in", "iwan", "ika", "ma", "kel", "sil"]

tt_pn_pred, v_pn_pred, pn_r = count_affix(tt_preds, v_preds, references, pronouns)
tt_adv_pred, v_adv_pred, adv_r = count_affix(tt_preds, v_preds, references, adverbs)
tt_prep_pred, v_prep_pred, prep_r = count_affix(tt_preds, v_preds, references, prepositions)

title = ["Pronoun", "Adverbial", "Prepositional"]
predictions = [[tt_pn_pred, v_pn_pred, pn_r], [tt_adv_pred, v_adv_pred, adv_r], [tt_prep_pred, v_prep_pred, prep_r]]
features = [pronouns, adverbs, prepositions]

for i in range(3):
    print("Trans-tokenization ", title[i], " counts:", predictions[i][0], "\nVs Vanilla ", title[i], " counts:", predictions[i][1], "\nVs. Reference ", title[i], " counts:", predictions[i][2], "\n")

# Plots
for i in range(3):
    tt_vals = [predictions[i][0].get(f, 0) for f in features[i]]
    v_vals = [predictions[i][1].get(f, 0) for f in features[i]]
    ref_vals = [predictions[i][2].get(f, 0) for f in features[i]]

    x = np.arange(len(features[i]))
    width = 0.25

    plt.figure(figsize=(10,6))
    plt.figure(figsize=(10,6))
    plt.bar(x - width, tt_vals, width, label="TT Predictions", color="red")
    plt.bar(x, v_vals, width, label="Vanilla Predictions", color="green")
    plt.bar(x + width, ref_vals, width, label="References", color="blue")

    plt.xticks(x, adverbs, rotation=45, ha="right")
    plt.ylabel("Frequency")
    plt.title(title[i], " Frequencies: TT vs Vanilla vs References")
    plt.legend()
    plt.tight_layout()
    plt.show()