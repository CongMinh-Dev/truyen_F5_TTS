from pedalboard import Pedalboard, Reverb, Chorus, Distortion, Bitcrush, HighpassFilter, Phaser, Gain
from pedalboard.io import AudioFile
from pydub import AudioSegment
import numpy as np
import os

def convert_voice_he_thong(input_path, background_path, output_path):
    # --- BƯỚC 1: XỬ LÝ HIỆU ỨNG GIỌNG NÓI (PEDALBOARD) ---
    with AudioFile(input_path) as f:
        samplerate = f.samplerate
        audio_data = f.read(f.frames)

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=450),
        Phaser(rate_hz=0.0, depth=0.8, feedback=0.6, centre_frequency_hz=1000), #càng lớn là càng giật giật
        Bitcrush(bit_depth=8), #càng lớn là càng rè
        Chorus(rate_hz=4.0, depth=0.3, feedback=0.3),
        Distortion(drive_db=3), #càng lớn là âm thanh giọng nói càng méo, nghe càng chói tai. ban đầu là 5
        Reverb(room_size=0.15, damping=0.4, wet_level=0.25, dry_level=0.75),
        Gain(gain_db=2) #càng lớn thì âm tổng thể càng to. ban đầ là 4
    ])

    effected = board(audio_data, samplerate)

    if effected.ndim == 1:
        effected = np.stack([effected, effected])
    elif effected.shape[0] == 1:
        effected = np.concatenate([effected, effected], axis=0)

    temp_voice = "temp_voice_pro.wav"
    with AudioFile(temp_voice, 'w', samplerate, 2) as o:
        o.write(effected)

    # --- BƯỚC 2: TRỘN NHẠC NỀN (PYDUB) ---
    voice_segment = AudioSegment.from_file(temp_voice)
    bg_segment = AudioSegment.from_file(background_path)

    # 1. Tăng âm lượng nhạc nền:
    bg_segment = bg_segment + 2

    # 2. Kéo dài nhạc nền cho khớp với giọng nói
    # Nếu nhạc nền ngắn hơn giọng nói, lặp lại nó cho đến khi dài hơn
    duration_voice = len(voice_segment)
    while len(bg_segment) < duration_voice:
        bg_segment += bg_segment
    
    # 3. Cắt nhạc nền để có độ dài CHÍNH XÁC bằng giọng nói
    bg_segment = bg_segment[:duration_voice]

    # 4. Trộn (Overlay)
    # Kết quả sẽ có độ dài đúng bằng voice_segment
    combined = voice_segment.overlay(bg_segment)

    # Xuất file
    combined = combined.fade_out(5)
    combined = combined.fade_in(5)

    combined.export(output_path, format="wav")
    
    # Xóa file tạm
    if os.path.exists(temp_voice):
        os.remove(temp_voice)
    
    print(f"Thành công! Độ dài file: {len(combined)/1000} giây")
    print(f"File lưu tại: {output_path}")

# Đường dẫn
input_voice = "/content/F5-TTS-2/infer_cli_basic_(2).wav"
background_sound = "/content/F5-TTS-2/sound_bg_he_thong.MP3"
output_file = "/content/F5-TTS-2/system_voice_final_with_bg.wav"

if __name__ == "__main__":
    # Chạy
    convert_voice_he_thong(input_voice, background_sound, output_file)