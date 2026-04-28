# 2
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset as HFDataset
import evaluate
from sklearn.model_selection import train_test_split
import re
import gradio as gr
import sys


import argparse
parser = argparse.ArgumentParser(
    description="train va xuat model add dau cau.",
)

parser.add_argument(
    "--path_dataset",
    type=str,
    help="path to dataset.json",
)
parser.add_argument(
    "--path_output",
    type=str,
    help="path to xuat model",
)
args = parser.parse_args()

path_dataset = args.path_dataset
path_output = args.path_output

if not path_dataset:
    print("chưa nhập path_dataset")
    sys.exit(1)

if not path_output:
    print("chưa nhập path_output")
    sys.exit(1)


# 3
with open(f"{path_dataset}", "r", encoding="utf-8") as f:
    data = json.load(f)

original_texts = [sample["original_text"] for sample in data]
corrected_texts = [sample["corrected_text"] for sample in data]

train_texts, test_texts, train_labels, test_labels = train_test_split(
    original_texts,
    corrected_texts,
    test_size=0.2,
    random_state=42
)

# 4
tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

def preprocess_function(example):
    model_inputs = tokenizer(example["input"], padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(example["output"], padding="max_length", truncation=True, max_length=256)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_data = [{"input": inp, "output": out} for inp, out in zip(train_texts, train_labels)]
test_data = [{"input": inp, "output": out} for inp, out in zip(test_texts, test_labels)]

train_dataset = HFDataset.from_list(train_data)
test_dataset = HFDataset.from_list(test_data)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)
# 5
model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable",use_safetensors=True)
# 6
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=7,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    # Thay logging_dir bằng đối số mới nếu bạn dùng TensorBoard
    logging_steps=100,
    gradient_accumulation_steps=1,
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    # Đổi tokenizer thành processing_class
    processing_class=tokenizer,
)

trainer.train()

# xuất model 
import os
os.makedirs(path_output, exist_ok=True)

model.save_pretrained(path_output)
tokenizer.save_pretrained(path_output)