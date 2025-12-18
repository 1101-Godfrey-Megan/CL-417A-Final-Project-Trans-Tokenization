import torch
import os
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pandas as pd
import evaluate

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


# Calculate chrF
chrf = evaluate.load("chrf")
tt_score = chrf.compute(predictions=tt_preds, references=references)
v_score = chrf.compute(predictions=v_preds, references=references)

print("Trans-Tokenization chrF Score:", tt_score)
print("Vanilla chrF Score:", v_score)