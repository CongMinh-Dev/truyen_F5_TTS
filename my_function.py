import os

def get_model_content(path, model_name):
    '''
    đưa vào <path>, <odel_name>. để khi đó nó sẽ vào đúng đường dẫn đó path/model_name/model_name.txt để lấy nội dung trong đó ra mà trả về
    '''

    # Kết hợp các thành phần để tạo thành đường dẫn: path/model_name/model_name.txt
    file_path = os.path.join(path, model_name, f"{model_name}.txt")

    try:
        # Mở và đọc nội dung file
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"Lỗi: Không tìm thấy file tại {file_path}"
    except Exception as e:
        return f"Đã xảy ra lỗi: {e}"


from pydub import AudioSegment
from pydub.silence import detect_leading_silence

def trim_silence(input_path, output_path, silence_threshold=-40.0, chunk_size=10, padding_ms=40):
    """
    Cắt bỏ khoảng lặng ở đầu và cuối nhưng giữ lại một khoảng đệm (padding) để âm thanh tự nhiên hơn.
    input_path và output_path được phép trùng nhau, nó sẽ ghi đè
    
    :param input_path: Đường dẫn file đầu vào.
    :param output_path: Đường dẫn file đầu ra.
    :param silence_threshold: Ngưỡng âm thanh (dBFS).
    :param chunk_size: Kích thước phân đoạn kiểm tra (ms).
    :param padding_ms: Khoảng lặng giữ lại (ms) ở đầu và cuối để tránh bị khựng.
    """
    try:
        # 1. Load file âm thanh
        audio = AudioSegment.from_file(input_path)
        duration = len(audio)

        # 2. Tìm điểm bắt đầu
        start_trim = detect_leading_silence(audio, silence_threshold, chunk_size)

        # 3. Tìm điểm kết thúc
        reversed_audio = audio.reverse()
        end_trim = detect_leading_silence(reversed_audio, silence_threshold, chunk_size)

        # --- ÁP DỤNG GIẢI PHÁP C: THÊM PADDING ---
        # Thay vì cắt sát, ta lùi điểm cắt ra một khoảng padding_ms
        # Sử dụng max và min để đảm bảo không cắt ngoài phạm vi độ dài file
        actual_start = max(0, start_trim - padding_ms)
        actual_end = duration - max(0, end_trim - padding_ms)

        # 4. Cắt file với khoảng đệm đã tính toán
        trimmed_audio = audio[actual_start:actual_end]

        # tạo âm thanh nhỏ xuống ở cuối file , lớn lên ở đầu file 1 cách từ từ
        trimmed_audio = trimmed_audio.fade_in(20).fade_out(20)

        # 5. Xuất file
        trimmed_audio.export(output_path, format="wav")
        
        print(f"✅ Xử lý thành công (đã giữ lại {padding_ms}ms đệm)!")
        print(f"   - Độ dài gốc: {duration/1000:.2f}s")
        print(f"   - Độ dài sau khi cắt: {len(trimmed_audio)/1000:.2f}s")

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

