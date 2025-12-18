import torch
import os
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import math

# Set Up
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Models, tokenizer, and create custom BOS for Nahuatl
base_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt").to(device)
v_model = MBartForConditionalGeneration.from_pretrained(os.path.join(os.getcwd(), "Vanilla")).to(device)
tt_model = MBartForConditionalGeneration.from_pretrained(os.path.join(os.getcwd(), "TransTokenized")).to(device)
tokenizer = MBart50TokenizerFast.from_pretrained(os.path.join(os.getcwd(), "Vanilla"))

tokenizer.src_lang = "es_XX"
if "nah_XX" not in tokenizer.lang_code_to_id:
    tokenizer.lang_code_to_id["nah_XX"] = len(tokenizer.lang_code_to_id)

# Load Dataset
dataset = load_dataset("wesamhaddad14/spanishNLP")

# Perplexity
def batched_perplexity(model, tokenizer, texts, batch_size=16, max_length=256):
    perps = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
        labels = enc["input_ids"].clone()
        with torch.no_grad():
            out = model(**enc, labels=labels)
        perps.append(math.exp(out.loss.item()))
    return sum(perps)/len(perps)

text = dataset["train"]["content"]

perp_base = batched_perplexity(base_model, tokenizer, text)
perp_v = batched_perplexity(v_model, tokenizer, text)
perp_tt = batched_perplexity(tt_model, tokenizer, text)

print("\nBase:", perp_base, "\nVanilla:", perp_v, "\nTrans-tokenization:", perp_tt, "\n")