@after_this_request
def remove_file(response):
    try:
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
            print(f"Đã xóa file tạm: {output_file_path}")
    except Exception as e:
        app.logger.error(f"Lỗi khi xóa file tạm: {e}")
    return response