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



