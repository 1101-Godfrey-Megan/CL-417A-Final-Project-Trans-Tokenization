import torch
import os
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Set Up
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = os.path.join(os.getcwd(), "Vanilla")

# Load Model, tokenizer, and create custom BOS for Nahuatl
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt").to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
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

# Unfreeze Only Top and Bottom 2 Layers, and LM Head Tied to Embedding Space
for p in model.parameters():
    p.requires_grad = False

for layer in model.model.encoder.layers[:2]:
    for p in layer.parameters():
        p.requires_grad = True

for layer in model.model.encoder.layers[-2:]:
    for p in layer.parameters():
        p.requires_grad = True

for layer in model.model.decoder.layers[:2]:
    for p in layer.parameters():
        p.requires_grad = True

for layer in model.model.decoder.layers[-2:]:
    for p in layer.parameters():
        p.requires_grad = True

for p in model.lm_head.parameters():
    p.requires_grad = True

# Training and Saving Model
training_args = Seq2SeqTrainingArguments(output_dir="./results", eval_strategy="epoch", learning_rate=2e-5, per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=4, num_train_epochs=18, predict_with_generate=False, fp16=True, report_to="none")
model.config.use_cache = False
trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"], tokenizer=tokenizer,)

trainer.train()
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)