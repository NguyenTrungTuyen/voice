from gtts import gTTS
import os

# Văn bản tiếng Việt
text = """
Top 15 cuốn sách nên đọc trong đời
Trăm năm cô đơn – Gabriel García Márquez
Vũ trụ - Carl Sagan
Lược sử vạn vật - Bill Bryson
Đắc nhân tâm - Dale Carnegie
Bá tước Monte Cristo - Alexandre Dumas
Những người khốn khổ - Victor Hugo
Những tấm lòng cao cả - Edmondo De Amicis
Mật mã Da Vinci - Dan Brown
"""

# Tạo giọng nói bằng Google TTS (tiếng Việt)
tts = gTTS(text, lang="vi", slow=False)  # `slow=False` để nói tốc độ bình thường

# Lưu file MP3
output_path = "output_google_tts.mp3"
tts.save(output_path)

# Phát file bằng trình phát mặc định (Windows)
os.system(f'start "" "{output_path}"')

print(f"✅ Đã tạo file giọng nói: {output_path}")