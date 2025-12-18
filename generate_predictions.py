import torch
import os
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pandas as pd
import tqdm

# Set Up
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = os.path.join(os.getcwd(), "Vanilla")

# Load Models, tokenizer, and create custom BOS for Nahuatl
v_model = MBartForConditionalGeneration.from_pretrained("./Vanilla")
tt_model = MBartForConditionalGeneration.from_pretrained("./TransTokenized")
tokenizer = MBart50TokenizerFast.from_pretrained("./Vanilla")

tokenizer.src_lang = "es_XX"
if "nah_XX" not in tokenizer.lang_code_to_id:
    tokenizer.lang_code_to_id["nah_XX"] = len(tokenizer.lang_code_to_id)

# Load and Preprocess Dataset
dataset = load_dataset("Thermostatic/Axolotl_Classical_Nahuatl_INALI")
def preprocess(batch):
    inputs = tokenizer(batch["sp"], max_length=256, truncation=True, padding="max_length")
    targets = tokenizer(batch["nah_inali"], max_length=256, truncation=True, padding="max_length")
    inputs["labels"] = [[(id if id != tokenizer.pad_token_id else -100) for id in seq]for seq in targets["input_ids"]]
    return inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# Make Predictions and Store Inputs, Prediction, and Labels
val_data = dataset["validation"]
batch_size = 16
tt_decoded_preds, v_decoded_preds, decoded_labels, spanish_inputs = [], [], [], []

for start in tqdm(range(0, len(val_data), batch_size), desc="Generating translations", unit="batch"):
    batch = val_data[start:start+batch_size]
    sp_texts = batch["sp"]
    labels = batch["nah_inali"]

    inputs = tokenizer(sp_texts, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
    tt_outputs = tt_model.generate(**inputs, max_length=256, num_beams=1, forced_bos_token_id=forced_bos_token_id)
    tt_preds = tokenizer.batch_decode(tt_outputs, skip_special_tokens=True)
    v_outputs = v_model.generate(**inputs, max_length=256, num_beams=1, forced_bos_token_id=forced_bos_token_id)
    v_preds = tokenizer.batch_decode(v_outputs, skip_special_tokens=True)

    tt_decoded_preds.extend(tt_preds)
    v_decoded_preds.extend(v_preds)
    decoded_labels.extend(labels)
    spanish_inputs.extend(sp_texts)

results = pd.DataFrame({
    "Spanish": spanish_inputs,
    "Vanilla Prediction": v_decoded_preds,
    "Trans-Tokenization Prediction": tt_decoded_preds,
    "Reference": decoded_labels
})
final_path = os.path.join(os.getcwd(), "validation_predictions.csv")
results.to_csv(final_path, index=False, encoding="utf-8")
print(f"Final predictions saved to {final_path}")