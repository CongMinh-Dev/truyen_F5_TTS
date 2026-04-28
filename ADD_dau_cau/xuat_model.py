output_dir = "/content/drive/MyDrive/model_add_dau_cau"
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
os.makedirs(output_dir, exist_ok=True)

model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable",use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)