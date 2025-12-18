import re
from datasets import load_dataset
from transformers import MBart50TokenizerFast

# Load Tokenizer
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer.src_lang = "es_XX"
if "nah_XX" not in tokenizer.lang_code_to_id:
    tokenizer.lang_code_to_id["nah_XX"] = len(tokenizer.lang_code_to_id)

# Load and Preprocess Dataset for FastAlign
def normalize_for_fastalign(text):
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

dataset = load_dataset("Thermostatic/Axolotl_Classical_Nahuatl_INALI")

with open("parallel.txt", "w", encoding="utf-8") as f:
    for sp, nah in zip(dataset["train"]["sp"], dataset["train"]["nah_inali"]):
        f.write(normalize_for_fastalign(sp) + " ||| " + normalize_for_fastalign(nah) + "\n")