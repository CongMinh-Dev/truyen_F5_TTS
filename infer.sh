f5-tts_infer-cli \
--model "F5TTS_v1_Base" \
--ref_audio ref.wav \
--ref_text "cả hai bên hãy cố gắng hiểu cho nhau" \
--gen_text "mình muốn ra nước ngoài để tiếp xúc nhiều công ty lớn, sau đó mang những gì học được về việt nam giúp xây dựng các công trình tốt hơn" \
--speed 1.0 \
--vocoder_name vocos \
--ckpt_file ckpts/model_last.pt \
# --output_dir  nếu không truyền thì nó tự tạo thư mục test ở cấp ngoài cùng, rồi lưu vào đó
# --vocab_file data/your_training_dataset/vocab.txt \
# --ckpt_file ckpts/your_training_dataset/model_last.pt \
