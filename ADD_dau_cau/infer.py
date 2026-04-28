import re
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

model_path = "/content/drive/MyDrive/model_add_dau_cau"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Ví dụ nhập một câu cụ thể
input_text = "sau khi quét thì thấy đây là phần quà cấp ét tên là giáng ngục phần quà là thanh gươm với thông tin chi tiết như sau sức mạnh ma pháp cộng chín trăm chín mươi chín không thể cường hóa thêm chỉ số đây là thanh gươm gieo rắc sự hủy diệt và hỗn loạn một ngày nào đó nó sẽ khiến địa ngục giáng trần"
result = correct_punctuation(input_text)

print("Kết quả sau khi xử lý:")
print(result)