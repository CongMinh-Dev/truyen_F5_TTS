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
