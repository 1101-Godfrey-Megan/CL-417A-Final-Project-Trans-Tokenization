import os
import re
import torch
from collections import defaultdict
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Set Up
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = os.path.join(os.getcwd(), "TransTokenized")

# Load Model, tokenizer, and create custom BOS for Nahuatl
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt").to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer.src_lang = "es_XX"
if "nah_XX" not in tokenizer.lang_code_to_id:
    tokenizer.lang_code_to_id["nah_XX"] = len(tokenizer.lang_code_to_id)

# Load and Preprocess Dataset
def preprocess(batch):
    inputs = tokenizer(batch["sp"], max_length=256, truncation=True, padding="max_length")
    targets = tokenizer(batch["nah_inali"], max_length=256, truncation=True, padding="max_length")
    inputs["labels"] = [[(id if id != tokenizer.pad_token_id else -100) for id in seq]for seq in targets["input_ids"]]
    return inputs

dataset = load_dataset("Thermostatic/Axolotl_Classical_Nahuatl_INALI")
tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)


# FastAlign Indice Probabilities to Token Probabilities
def split_like_fa(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

nah_to_span_dict = defaultdict(lambda:defaultdict(int))

with open("parallel.txt", "r", encoding="utf-8") as text_file, open("token_relations.txt", "r", encoding="utf-8") as alignments_file:
    for text_line, aligments_line in zip(text_file, alignments_file):
        src_sentence, tgt_sentence = text_line.strip().split(" ||| ")
        src_words = split_like_fa(src_sentence.strip())
        tgt_words = split_like_fa(tgt_sentence.strip())

        for pair in aligments_line.strip().split():
            src_idx, tgt_idx = map(int, pair.split("-"))

            if src_idx < len(src_words) and tgt_idx < len(tgt_words):
                src_word = src_words[src_idx]
                tgt_word = tgt_words[tgt_idx]
                nah_to_span_dict[src_word][tgt_word] += 1

            else:
                pass

nah_alignment_dict = defaultdict(lambda:defaultdict(int))

for nah_word, span_dict in nah_to_span_dict.items():
    for span_word, count in span_dict.items():
        if count < 10:
            continue
        else:
            span_tokens = tokenizer.tokenize(span_word)
            nah_tokens = tokenizer.tokenize(nah_word)
            span_id = tokenizer.convert_tokens_to_ids(span_tokens)
            nah_id  = tokenizer.convert_tokens_to_ids(nah_tokens)

            # One to One Mapping
            if len(span_tokens) == 1 and len(nah_tokens) == 1:
                nah_alignment_dict[nah_id[0]][span_id[0]] += count #might need to remove [0]
            # Many to One Mapping
            elif (len(span_tokens) > len(nah_tokens)) and len(nah_tokens) == 1:
                for s in span_id:
                    nah_alignment_dict[nah_id[0]][s] += (count/len(span_tokens))
            # One to Many Mapping
            elif (len(nah_tokens) > len(span_tokens)) and len(span_tokens) == 1:
                for n in nah_id:
                    nah_alignment_dict[n][span_id[0]] += (count/len(nah_tokens))
            # All-to-all and In-order Mapping
            elif (len(nah_tokens) == len(span_tokens)) and len(nah_tokens) >1:
                for n in range(len(nah_id)):
                    for s in span_id:
                        nah_alignment_dict[nah_id[n]][s] += ((0.5* count) / (len(nah_tokens) * len(span_tokens)))
                        nah_alignment_dict[nah_id[n]][span_id[n]] += ((0.5 * count) / (len(nah_tokens)))

            # All-to-all and Proportional In-order Mapping
            else:
                for n in nah_id:
                    for s in span_id:
                        nah_alignment_dict[n][s] += ((0.5 * count) / (len(span_tokens) * len(nah_tokens)))

                span_intervals = [(i/len(span_tokens), (i+1)/len(span_tokens)) for i in range(len(span_tokens))]
                nah_intervals = [(j/(len(nah_tokens)*len(nah_tokens)), (j+1)/(len(nah_tokens)*len(nah_tokens))) for j in range(len(nah_tokens)*len(nah_tokens))]

                for j in range(len(nah_tokens)):
                    total_overlap = {i: 0 for i in range(len(span_tokens))}
                    for k in range(len(nah_tokens)):
                        n_start, n_end = nah_intervals[j+k*len(nah_tokens)]
                        for i, (s_start, s_end) in enumerate(span_intervals):
                            overlap = max(0, min(n_end, s_end) - max(n_start, s_start))
                            if overlap > 0:
                                total_overlap[i] += overlap
                    for i in range(len(span_tokens)):
                        nah_alignment_dict[nah_id[j]][span_id[i]] += ((total_overlap[i] * 0.5 * count) / (len(span_tokens)))

# Align Embedding Mappings
embeddings = model.model.shared.weight

for n_id, sp_dict in nah_alignment_dict.items():
    total = sum(sp_dict.values())
    if total == 0:
        continue
    else:
        weighted_vec = torch.zeros_like(embeddings[n_id])
        for sp_id, w in sp_dict.items():
            prob = w / total
            weighted_vec += prob * embeddings[sp_id]
        with torch.no_grad():
            embeddings[n_id] = weighted_vec

with torch.no_grad():
    tt = embeddings[n_id]
    norm = tt.norm(p=2)
    if norm > 0:
        embeddings[n_id] = tt / norm


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
training_args = Seq2SeqTrainingArguments(output_dir="./results", eval_strategy="epoch", learning_rate=2e-5, per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=4, num_train_epochs=19, predict_with_generate=False, fp16=True, report_to="none")
model.config.use_cache = False
trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"], tokenizer=tokenizer,)

trainer.train()
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)