import sys
import os
import torch
import numpy as np
import soundfile as sf
from flask import Flask, request, send_file, jsonify, after_this_request
from flask_cors import CORS
from omegaconf import OmegaConf
from hydra.utils import get_class
#  Thêm thư mục gốc (nơi chứa các file convert_voice_*.py)
sys.path.insert(0, "/content/F5-TTS-2")

# Thêm đường dẫn src để nhận diện module f5_tts
sys.path.insert(0, "/content/F5-TTS-2/src")

# Import core từ thư viện F5-TTS
from f5_tts.infer.utils_infer import (
    load_model, load_vocoder, infer_process, device
)

# Import các tool hậu xử lý của bạn
from convert_voice_suy_nghi import convert_voice_suy_nghi
from convert_voice_he_thong import convert_voice_he_thong
from convert_voice_loa_phat_thanh import convert_voice_loa_phat_thanh

app = Flask(__name__)
CORS(app)

# --- CẤU HÌNH HỆ THỐNG ---
MODEL_NAME = "F5TTS_Base"
CKPT_FILE = "/content/F5-TTS-2/ckpts/model_last.pt"
VOCAB_FILE = "/content/F5-TTS-2/data/Emilia_ZH_EN_pinyin/vocab.txt"
VOCODER_NAME = "vocos"
OUTPUT_DIR = "/content/F5-TTS-2/output_audio"
BG_HE_THONG = "/content/F5-TTS-2/model_name_mp3/sound_bg_he_thong.MP3"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- BƯỚC 1: NẠP MODEL (CHỈ CHẠY 1 LẦN KHI START) ---
print(f"--- Đang nạp model vào {device} ---")
vocoder = load_vocoder(vocoder_name=VOCODER_NAME, is_local=False, device=device)

# Load config từ thư mục src
model_cfg_path = f"/content/F5-TTS-2/src/f5_tts/configs/{MODEL_NAME}.yaml"
model_cfg = OmegaConf.load(model_cfg_path)
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch

ema_model = load_model(
    model_cls, model_arc, CKPT_FILE, 
    mel_spec_type=VOCODER_NAME, 
    vocab_file=VOCAB_FILE, 
    device=device
)
print("✅ Model đã nạp thành công và sẵn sàng trên GPU!")

# Mapping ref_text
ref_texts = {
    "nam_chinh": "nội dung của file ref nam chính",
    "gai_tre6": "nội dung của file ref gái trẻ",
    "default": "cả hai bên hãy cố gắng hiểu cho nhau"
}

# --- BƯỚC 2: ĐỊNH NGHĨA API ---
@app.route('/convert_text', methods=['POST'])
def convert_text():
    try:
        data = request.get_json()
        text = data.get('text')
        model_name = data.get('model_name')
        file_name = data.get('fileName', 'output')
        toc_do_doc = float(data.get('toc_do_doc', 1.0))
        suy_nghi_val = int(data.get('suy_nghi', 0))

        # Đường dẫn file MP3 mẫu
        ref_audio_path = f"/content/F5-TTS-2/model_name_mp3/{model_name}/{model_name}.MP3"
        if not os.path.exists(ref_audio_path):
            return jsonify({"error": f"Không tìm thấy: {ref_audio_path}"}), 400

        current_ref_text = ref_texts.get(model_name, ref_texts["default"])

        # Thực hiện TTS
        audio_segment, final_sample_rate, _ = infer_process(
            ref_audio_path, current_ref_text, text,
            ema_model, vocoder, mel_spec_type=VOCODER_NAME,
            speed=toc_do_doc, device=device
        )

        temp_raw_path = os.path.join(OUTPUT_DIR, f"raw_{file_name}.wav")
        final_output_path = os.path.join(OUTPUT_DIR, f"{file_name}.wav")
        sf.write(temp_raw_path, audio_segment, final_sample_rate)

        # Hậu xử lý hiệu ứng
        if suy_nghi_val != 0:
            convert_voice_suy_nghi(temp_raw_path, final_output_path, pan=-0.15)
        elif model_name == "hop_thoai_may":
            convert_voice_he_thong(temp_raw_path, BG_HE_THONG, final_output_path)
        elif model_name == "loa_phat_thanh":
            # background_path để trống nếu không dùng nền cho loa phường
            convert_voice_loa_phat_thanh(temp_raw_path, "", final_output_path)
        else:
            if os.path.exists(final_output_path): os.remove(final_output_path)
            os.rename(temp_raw_path, final_output_path)

        if os.path.exists(temp_raw_path): os.remove(temp_raw_path)

        @after_this_request
        def remove_file(response):
            try:
                if os.path.exists(final_output_path): os.remove(final_output_path)
            except: pass
            return response

        return send_file(final_output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Chạy Flask ở port 5555
    app.run(host='0.0.0.0', port=5555)