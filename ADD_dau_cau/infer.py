import re
import argparse
parser = argparse.ArgumentParser(
    description="train va xuat model add dau cau.",
)

parser.add_argument(
    "--model_path",
    type=str,
    help="path to thư mục chứa model",
)
parser.add_argument(
    "--input_text",
    type=str,
    help="nội dung cần thêm dấu",
)

args = parser.parse_args()
model_path = args.model_path
input_text = args.input_text

if not model_path:
    print("chưa nhập model_path")
    sys.exit(1)

if not input_text:
    print("chưa nhập input_text")
    sys.exit(1)


def fix_spacing(text):
    text = re.sub(r'\s+([.,!?:;])', r'\1', text)
    text = re.sub(r'([.,!?:;])(?=\S)', r'\1 ', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\"\s+', '"', text)
    text = re.sub(r'\s+\"', '"', text)
    return text.strip()

def separate_words_and_punctuations(text):
    tokens = re.findall(r"\w+|[.,;:?!…'\"]+", text, flags=re.UNICODE)
    return " ".join(tokens)

def correct_punctuation(text):
    text = separate_words_and_punctuations(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    outputs = model.generate(**inputs, max_length=128, num_beams=4)
    return fix_spacing(tokenizer.decode(outputs[0], skip_special_tokens=True))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Ví dụ nhập một câu cụ thể

result = correct_punctuation(input_text)

print("Kết quả sau khi xử lý:")
print(result)