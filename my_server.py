import sys
import os
import torch
import numpy as np
import soundfile as sf
import re
from flask import Flask, request, send_file, jsonify, after_this_request
from flask_cors import CORS
from omegaconf import OmegaConf
from hydra.utils import get_class
from pathlib import Path

# Thêm đường dẫn
sys.path.insert(0, "/content/F5-TTS-2")
sys.path.insert(0, "/content/F5-TTS-2/src")

from f5_tts.infer.utils_infer import (
    load_model, 
    load_vocoder, 
    infer_process, 
    device, 
    preprocess_ref_audio_text,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    fix_duration,
)
from my_function import get_model_content , trim_silence
from convert_voice_loa_phat_thanh import convert_voice_loa_phat_thanh
from convert_voice_he_thong import convert_voice_he_thong
from convert_voice_suy_nghi import convert_voice_suy_nghi



app = Flask(__name__)
CORS(app)

# --- CẤU HÌNH HỆ THỐNG ---
MODEL= "F5TTS_Base" #nó là tên file yaml chứ không phải tên MODEL.
CKPT_FILE = "/content/F5-TTS-2/ckpts/model_last.pt"
VOCAB_FILE = "/content/F5-TTS-2/data/Emilia_ZH_EN_pinyin/vocab.txt"
vocoder_name = "vocos"
OUTPUT_DIR = "/content/F5-TTS-2/output_audio"
MP3_DIR = "/content/F5-TTS-2/model_name_mp3"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- BƯỚC 1: NẠP MODEL (CHẠY 1 LẦN) ---
print(f"--- Đang nạp model vào {device} ---")
# load vocoder
vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, device=device)

# load model
model_cfg_path = f"/content/F5-TTS-2/src/f5_tts/configs/{MODEL}.yaml"
model_cfg = OmegaConf.load(model_cfg_path)
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch
ema_model = load_model(
    model_cls, model_arc, CKPT_FILE,
    mel_spec_type=vocoder_name,
    vocab_file=VOCAB_FILE,
    device=device
)
print("✅ Model đã nạp thành công!")

@app.route('/convert_text', methods=['POST'])
def convert_text():
    try:
        data = request.get_json()
        gen_text = data.get('text')
        model_name = data.get('model_name')
        file_name = f"{data.get('fileName', 'output')}.wav"
        # toc_do_doc = float(data.get('toc_do_doc', 1.0))
        toc_do_doc = float(0.7)
        suy_nghi = int(data.get('suy_nghi', 0))
        character = data.get('character')


        ref_audio_path = f"{MP3_DIR}/{model_name}/{model_name}.mp3"
        if not os.path.exists(ref_audio_path):
            return jsonify({"error": f"Không tìm thấy audio mẫu: {ref_audio_path} tại {file_name}"}), 400


        # --- QUAN TRỌNG: XỬ LÝ GENERATOR ---
        wave_path = str(Path(OUTPUT_DIR) / file_name) #vì path trả  về PosixPath (Một đối tượng/object). nên cần biến nó thành string
        wave_path_tam_thoi = os.path.join(OUTPUT_DIR, f"tam_{file_name}")
        ref_text = get_model_content(MP3_DIR, model_name)
        ref_audio = os.path.join(MP3_DIR, model_name, f"{model_name}.mp3")

        main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
        voices = {"main": main_voice}
        for voice in voices:
            print("Voice:", voice)
            print("ref_audio ", voices[voice]["ref_audio"])
            voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
            print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

        generated_audio_segments = []

        reg1 = r"(?=\[\w+\])"
        reg2 = r"\[(\w+)\]"
        chunks = re.split(reg1, gen_text)
        for text in chunks:
            if not text.strip():
                continue
            match = re.match(reg2, text)
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text = re.sub(reg2, "", text)
            ref_audio_ = voices[voice]["ref_audio"]
            ref_text_ = voices[voice]["ref_text"]
            local_speed = voices[voice].get("speed", toc_do_doc)
            gen_text_ = text.strip()
            print(f"Voice: {voice}")
            audio_segment, final_sample_rate, spectrogram = infer_process(
                ref_audio_,
                ref_text_,
                gen_text_,
                ema_model,
                vocoder,
                mel_spec_type=vocoder_name,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=local_speed,
                fix_duration=fix_duration,
                device=device,
            )
            generated_audio_segments.append(audio_segment)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)

            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

            with open(wave_path_tam_thoi, "wb") as f: #open() mở file hoặc tạo mới file
                #f.name chình là wave_path hay gọi là đường dẫn (vd: /content/output/mot.wav)
                #final_wave là nội dung file âm thanh
                #final_sample_rate hình như là 24khz, xem thêm bên utils_infer.py
                sf.write(f.name, final_wave, final_sample_rate) 
                print(f.name)

        # -----------------------------------


        # Hậu xử lý
        trim_silence(wave_path_tam_thoi, wave_path_tam_thoi)

        if suy_nghi != 0:
            convert_voice_suy_nghi(wave_path_tam_thoi, wave_path, pan=-0.15)
        elif character == "hop_thoai_may":
            convert_voice_he_thong(wave_path_tam_thoi, f"{MP3_DIR}/sound_bg_he_thong.mp3", wave_path)
        elif character == "loa_phat_thanh":
            convert_voice_loa_phat_thanh(wave_path_tam_thoi, "", wave_path)
        else:
            if os.path.exists(wave_path): os.remove(wave_path)
            os.rename(wave_path_tam_thoi, wave_path)

        if os.path.exists(wave_path_tam_thoi): os.remove(wave_path_tam_thoi)

        @after_this_request
        def remove_file(response):
            try:
                if os.path.exists(wave_path): os.remove(wave_path)
            except: pass
            return response

        return send_file(wave_path, as_attachment=True)
        

    except Exception as e:
        print(f"Lỗi: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555)