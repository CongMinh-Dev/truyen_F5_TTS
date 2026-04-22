import sys
import os

# Ép dùng đúng thư mục src của dự án
sys.path.insert(0, "/content/F5-TTS-2/src")

import torch
import soundfile as sf
from flask import Flask, request, send_file, jsonify, after_this_request
from flask_cors import CORS
from omegaconf import OmegaConf
from hydra.utils import get_class

# Import từ src/f5_tts
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process, device

# Import các hàm hậu xử lý của bạn
from convert_voice_suy_nghi import convert_voice_suy_nghi
from convert_voice_he_thong import convert_voice_he_thong
from convert_voice_loa_phat_thanh import convert_voice_loa_phat_thanh

app = Flask(__name__)
CORS(app)

# --- CẤU HÌNH MODEL ---
MODEL_NAME = "F5TTS_Base"
CKPT_FILE = "/content/F5-TTS-2/ckpts/model_last.pt"
VOCAB_FILE = "/content/F5-TTS-2/data/Emilia_ZH_EN_pinyin/vocab.txt"
OUTPUT_DIR = "/content/F5-TTS-2/output_audio"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

print(f"--- Đang nạp model vào {device} ---")
vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
model_cfg = OmegaConf.load(f"/content/F5-TTS-2/src/f5_tts/configs/{MODEL_NAME}.yaml")
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
ema_model = load_model(model_cls, model_cfg.model.arch, CKPT_FILE, vocab_file=VOCAB_FILE, device=device)
print("✅ Model đã sẵn sàng!")

ref_texts = {
    "nam_chinh": "cả hai bên hãy cố gắng hiểu cho nhau",
    "default": "cả hai bên hãy cố gắng hiểu cho nhau"
}

@app.route('/convert_text', methods=['POST'])
def convert():
    try:
        data = request.json
        text = data.get('text')
        model_name = data.get('model_name')
        file_name = data.get('fileName', 'output')
        suy_nghi_val = data.get('suy_nghi', 0)

        ref_audio_path = f"/content/F5-TTS-2/{model_name}/{model_name}.MP3"
        current_ref_text = ref_texts.get(model_name, ref_texts["default"])

        # Bước 1: TTS (Dùng trực tiếp MP3 của bạn)
        audio_segment, final_sample_rate, _ = infer_process(
            ref_audio_path, current_ref_text, text, 
            ema_model, vocoder, speed=float(data.get('toc_do_doc', 1.0)), device=device
        )

        temp_path = os.path.join(OUTPUT_DIR, f"raw_{file_name}.wav")
        final_path = os.path.join(OUTPUT_DIR, f"{file_name}.wav")
        sf.write(temp_path, audio_segment, final_sample_rate)

        # Bước 2: Hậu xử lý
        if suy_nghi_val != 0:
            convert_voice_suy_nghi(temp_path, final_path)
        elif model_name == "hop_thoai_may":
            convert_voice_he_thong(temp_path, "/content/F5-TTS-2/model_name_mp3/sound_bg_he_thong.MP3", final_path)
        else:
            os.rename(temp_path, final_path)

        @after_this_request
        def cleanup(response):
            if os.path.exists(final_path): os.remove(final_path)
            if os.path.exists(temp_path): os.remove(temp_path)
            return response

        return send_file(final_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5555)