import torch
import os
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pandas as pd

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


# Repetitions
def find_repetitions(text, min_repeat=3):
    tokens = text.split()
    reps = []
    count = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            count += 1
        else:
            if count > min_repeat:
                reps.append((tokens[i-1], count))
            count = 1
    if count > min_repeat:
        reps.append((tokens[-1], count))
    return reps

def repetition_count(preds, refs):
    total = 0
    details = []
    for pred, ref in zip(preds, refs):
        reps = find_repetitions(pred)
        for token, count in reps:
            if token not in ref.split():
                total += 1
                details.append((token, count, pred))
    return total, details

tt_total, tt_details = repetition_count(tt_preds, references)
v_total, v_details = repetition_count(v_preds, references)

print("Trans-tokenization repetitions:", tt_total)
print("Vanilla mBART repetitions:", v_total)
