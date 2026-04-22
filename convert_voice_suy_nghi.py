from pedalboard import Pedalboard, Reverb, Chorus, LowpassFilter, Gain, HighpassFilter
from pedalboard.io import AudioFile
import numpy as np

def convert_voice_suy_nghi(input_path, output_path, pan=-0.15):
    with AudioFile(input_path) as f:
        samplerate = f.samplerate
        audio_data = f.read(f.frames)

    # 1. Chạy chuỗi hiệu ứng làm dày và vang giọng
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=150), 
        LowpassFilter(cutoff_frequency_hz=1600), #càng giảm cái này thì giọng nge sẽ càng nghẹt mũi
        Chorus(rate_hz=1.0, depth=0.5, centre_delay_ms=10.0, feedback=0.4),
        Reverb(room_size=0.5, damping=0.5, wet_level=0.35, dry_level=0.7),
        Gain(gain_db=3) # 3 thì độ lớn bằng file gốc, mình test thấy -5 thì nhỏ xuống phù hợp cho giọng suy nghĩ
    ])

    effected = board(audio_data, samplerate)

    # 2. XỬ LÝ PANNING THỦ CÔNG (Đảm bảo Stereo và vị trí âm thanh)
    # Nếu là Mono (1 dòng), nhân bản thành 2 dòng (Trái, Phải)
    if effected.ndim == 1:
        effected = np.stack([effected, effected])
    elif effected.shape[0] == 1:
        effected = np.concatenate([effected, effected], axis=0)

    # pan = -1.0 (Trái hoàn toàn), 0.0 (Giữa), 1.0 (Phải hoàn toàn)
    # Công thức tính hệ số âm lượng cho 2 loa:
    left_gain = 1.0 - pan if pan > 0 else 1.0
    right_gain = 1.0 + pan if pan < 0 else 1.0
    
    # Áp dụng Panning vào mảng dữ liệu
    effected[0] *= left_gain  # Kênh trái
    effected[1] *= right_gain # Kênh phải

    # 3. Xuất file Stereo
    with AudioFile(output_path, 'w', samplerate, 2) as o:
        o.write(effected)
    
    print(f"Đã hoàn thành bản Final với Panning = {pan} tại: {output_path}")

# Sử dụng: pan=-0.15 giúp giọng hơi lệch sang tai trái một chút, nghe rất 'nội tâm'
convert_voice_suy_nghi("/content/F5-TTS-2/infer_cli_basic_(2).wav", "/content/F5-TTS-2/infer_cli_basic_output_pro.wav", pan=-0.15)