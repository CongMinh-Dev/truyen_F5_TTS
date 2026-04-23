import sys
import os
import torch
import numpy as np
import soundfile as sf
from flask import Flask, request, send_file, jsonify, after_this_request
from flask_cors import CORS
from omegaconf import OmegaConf
from hydra.utils import get_class

# Thêm đường dẫn
sys.path.insert(0, "/content/F5-TTS-2")
sys.path.insert(0, "/content/F5-TTS-2/src")

from f5_tts.infer.utils_infer import (
    load_model, load_vocoder, infer_process, device
)
from my_function import get_model_content
from convert_voice_loa_phat_thanh import convert_voice_loa_phat_thanh
from convert_voice_he_thong import convert_voice_he_thong
from convert_voice_suy_nghi import convert_voice_suy_nghi

app = Flask(__name__)
CORS(app)

# --- CẤU HÌNH HỆ THỐNG ---
MODEL_NAME = "F5TTS_Base"
CKPT_FILE = "/content/F5-TTS-2/ckpts/model_last.pt"
VOCAB_FILE = "/content/F5-TTS-2/data/Emilia_ZH_EN_pinyin/vocab.txt"
VOCODER_NAME = "vocos"
OUTPUT_DIR = "/content/F5-TTS-2/output_audio"
MP3_DIR = "/content/F5-TTS-2/model_name_mp3"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- BƯỚC 1: NẠP MODEL (CHẠY 1 LẦN) ---
print(f"--- Đang nạp model vào {device} ---")
vocoder = load_vocoder(vocoder_name=VOCODER_NAME, is_local=False, device=device)

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
print("✅ Model đã nạp thành công!")

@app.route('/convert_text', methods=['POST'])
def convert_text():
    try:
        data = request.get_json()
        text = data.get('text')
        model_name = data.get('model_name')
        file_name = data.get('fileName', 'output')
        toc_do_doc = float(data.get('toc_do_doc', 1.0))
        suy_nghi_val = int(data.get('suy_nghi', 0))

        ref_audio_path = f"{MP3_DIR}/{model_name}/{model_name}.MP3"
        if not os.path.exists(ref_audio_path):
            return jsonify({"error": f"Không tìm thấy audio mẫu: {ref_audio_path}"}), 400

        current_ref_text = get_model_content(MP3_DIR, model_name)

        # --- QUAN TRỌNG: XỬ LÝ GENERATOR ---
        # Lặp qua generator để lấy toàn bộ các đoạn âm thanh (batches)
        generated_segments = []
        final_sr = 24000 # Mặc định của F5-TTS

        # Gọi infer_process và hứng toàn bộ kết quả yield
        for audio_chunk, sr, _ in infer_process(
            ref_audio_path, current_ref_text, text,
            ema_model, vocoder, mel_spec_type=VOCODER_NAME,
            speed=toc_do_doc, device=device
        ):
            if audio_chunk is not None:
                generated_segments.append(audio_chunk)
                final_sr = sr

        if not generated_segments:
            return jsonify({"error": "Không thể tạo âm thanh (kết quả rỗng)"}), 500

        # Nối tất cả các đoạn âm thanh lại thành một mảng duy nhất
        audio_segment = np.concatenate(generated_segments)
        # -----------------------------------

        temp_raw_path = os.path.join(OUTPUT_DIR, f"raw_{file_name}.wav")
        final_output_path = os.path.join(OUTPUT_DIR, f"{file_name}.wav")
        
        # Ghi file từ mảng đã nối
        sf.write(temp_raw_path, audio_segment, final_sr)

        # Hậu xử lý
        if suy_nghi_val != 0:
            convert_voice_suy_nghi(temp_raw_path, final_output_path, pan=-0.15)
        elif model_name == "hop_thoai_may":
            convert_voice_he_thong(temp_raw_path, f"{MP3_DIR}/sound_bg_he_thong.MP3", final_output_path)
        elif model_name == "loa_phat_thanh":
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
        print(f"Lỗi: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555)