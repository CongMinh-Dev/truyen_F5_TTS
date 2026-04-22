from pedalboard import Pedalboard, Reverb, Delay, Distortion, HighpassFilter, LowpassFilter, Gain
from pedalboard.io import AudioFile
from pydub import AudioSegment
import numpy as np
import os

def convert_voice_loa_phat_thanh(input_path, background_path, output_path):
    # --- BƯỚC 1: XỬ LÝ GIỌNG ĐỌC (MÔ PHỎNG LOA PHƯỜNG/LOA THÀNH PHỐ) ---
    with AudioFile(input_path) as f:
        samplerate = f.samplerate
        audio_data = f.read(f.frames)

    board = Pedalboard([
        # 1. Ép dải tần: Cắt mạnh âm trầm để giọng "ông ổng"
        HighpassFilter(cutoff_frequency_hz=600),
        LowpassFilter(cutoff_frequency_hz=3500),
        
        # 2. Distortion: Tạo độ rè kim loại đặc trưng của loa công suất lớn
        Distortion(drive_db=10),
        
        # 3. Delay (Tiếng vọng): Đây là linh hồn của file mẫu bạn gửi
        # Tạo tiếng nhại lại rõ rệt như âm thanh đập vào các tòa nhà
        Delay(delay_seconds=0.25, feedback=0.3, mix=0.2),
        
        # 4. Reverb: Tạo không gian rộng lớn ngoài trời
        Reverb(room_size=0.6, damping=0.4, wet_level=0.25, dry_level=0.75),
        
        # 5. Gain: Đẩy âm lượng lên cao
        Gain(gain_db=6)
    ])

    effected = board(audio_data, samplerate)

    # Đảm bảo Stereo
    if effected.ndim == 1:
        effected = np.stack([effected, effected])
    elif effected.shape[0] == 1:
        effected = np.concatenate([effected, effected], axis=0)

    temp_voice = "temp_megaphone.wav"
    with AudioFile(temp_voice, 'w', samplerate, 2) as o:
        o.write(effected)

    # --- BƯỚC 2: TRỘN NHẠC NỀN (NẾU CÓ) ---
    voice_segment = AudioSegment.from_file(temp_voice)
    
    if os.path.exists(background_path):
        bg_segment = AudioSegment.from_file(background_path)
        
        # Chỉnh âm lượng nền (để tầm -5dB đến 0dB nếu muốn nghe rõ tiếng 0101 hoặc rè)
        bg_segment = bg_segment - 5 

        # Kéo dài nhạc nền cho khớp với giọng nói
        duration_voice = len(voice_segment)
        while len(bg_segment) < duration_voice:
            bg_segment += bg_segment
        bg_segment = bg_segment[:duration_voice]

        # Trộn nhạc nền vào giọng nói
        combined = voice_segment.overlay(bg_segment)
    else:
        combined = voice_segment

    # Xuất file cuối cùng
    combined.export(output_path, format="wav")
    
    # Xóa file tạm
    if os.path.exists(temp_voice):
        os.remove(temp_voice)
    
    print(f"Hoàn thành hiệu ứng loa thông báo! File lưu tại: {output_path}")

# Đường dẫn
input_voice = "/content/F5-TTS-2/infer_cli_basic_(2).wav"
background_sound = "" # Để rỗng nếu không cần nền
output_file = "/content/F5-TTS-2/loa_phat_thanh_hoan_thien.wav"

# Chạy
convert_voice_loa_phat_thanh(input_voice, background_sound, output_file)