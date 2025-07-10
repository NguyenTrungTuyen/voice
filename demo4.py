from TTS.api import TTS
import os
import librosa
import numpy as np
from pydub import AudioSegment
import warnings
warnings.filterwarnings("ignore")

class VietnameseTTS:
    def __init__(self):
        # Sử dụng model XTTS v2 - hỗ trợ voice cloning và multilingual
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        print(f"🔄 Đang tải model: {self.model_name}")
        
        try:
            self.tts = TTS(model_name=self.model_name, progress_bar=True, gpu=False)
            print("✅ Model đã được tải thành công!")
        except Exception as e:
            print(f"❌ Lỗi khi tải model: {e}")
            # Fallback sang model khác
            self.model_name = "tts_models/en/ljspeech/vits"
            print(f"🔄 Thử model backup: {self.model_name}")
            self.tts = TTS(model_name=self.model_name, progress_bar=True, gpu=False)
    
    def preprocess_audio(self, audio_path, target_sr=22050):
        """Tiền xử lý file audio để tối ưu cho voice cloning"""
        try:
            # Đọc file audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Chuyển về mono nếu cần
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            
            # Resample về target sample rate
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            # Trim silence
            y, _ = librosa.effects.trim(y, top_db=20)
            
            # Lưu file đã xử lý
            processed_path = audio_path.replace('.wav', '_processed.wav')
            import soundfile as sf
            sf.write(processed_path, y, target_sr)
            
            return processed_path, True
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý audio: {e}")
            return audio_path, False
    
    def check_wav_file(self, file_path):
        """Kiểm tra tính hợp lệ của file WAV"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ File không tồn tại: {file_path}")
                return False
                
            y, sr = librosa.load(file_path, sr=None)
            
            # Kiểm tra định dạng
            if not file_path.lower().endswith('.wav'):
                print("⚠️ File không phải định dạng .WAV")
                return False
            
            # Kiểm tra độ dài (ít nhất 3 giây, tối đa 30 giây cho voice cloning)
            duration = len(y) / sr
            if duration < 3:
                print(f"⚠️ File quá ngắn ({duration:.1f}s). Cần ít nhất 3 giây.")
                return False
            elif duration > 30:
                print(f"⚠️ File quá dài ({duration:.1f}s). Nên dưới 30 giây.")
            
            print(f"✅ File hợp lệ: {duration:.1f}s, {sr}Hz")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi kiểm tra file: {e}")
            return False
    
    def text_to_speech(self, text, voice_path=None, output_path="output.wav", language="vi"):
        """Chuyển text thành speech với voice cloning"""
        try:
            print(f"📝 Text: {text[:50]}...")
            
            if voice_path and os.path.exists(voice_path):
                if self.check_wav_file(voice_path):
                    # Tiền xử lý audio
                    processed_voice, success = self.preprocess_audio(voice_path)
                    
                    if success:
                        print(f"🔊 Đang clone giọng từ: {voice_path}")
                        
                        # Sử dụng XTTS v2 với voice cloning
                        if "xtts" in self.model_name:
                            self.tts.tts_to_file(
                                text=text,
                                speaker_wav=processed_voice,
                                language=language,
                                file_path=output_path
                            )
                        else:
                            # Fallback cho các model khác
                            self.tts.tts_to_file(
                                text=text,
                                speaker_wav=processed_voice,
                                file_path=output_path
                            )
                    else:
                        print("⚠️ Sử dụng giọng mặc định do lỗi xử lý file.")
                        self.tts.tts_to_file(text=text, file_path=output_path)
                else:
                    print("⚠️ File giọng không hợp lệ, sử dụng giọng mặc định.")
                    self.tts.tts_to_file(text=text, file_path=output_path)
            else:
                print("⚠️ Không có file giọng, sử dụng giọng mặc định.")
                self.tts.tts_to_file(text=text, file_path=output_path)
            
            if os.path.exists(output_path):
                print(f"✅ Đã tạo file: {output_path}")
                return True
            else:
                print("❌ Không thể tạo file output")
                return False
                
        except Exception as e:
            print(f"❌ Lỗi khi tạo speech: {e}")
            return False

def main():
    # Khởi tạo TTS
    vn_tts = VietnameseTTS()
    
    # Văn bản tiếng Việt
    text = """
    Top 15 cuốn sách nên đọc trong đời:
    Trăm năm cô đơn – Gabriel García Márquez
    Vũ trụ - Carl Sagan
    Lược sử vạn vật - Bill Bryson
    Đắc nhân tâm - Dale Carnegie
    Bá tước Monte Cristo - Alexandre Dumas
    """
    
    # Đường dẫn file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    voice_path = os.path.join(base_dir, "dyi.wav")
    output_path = os.path.join(base_dir, "output_vietnamese.wav")
    
    # Tạo speech
    success = vn_tts.text_to_speech(
        text=text,
        voice_path=voice_path,
        output_path=output_path,
        language="vi"
    )
    
    if success:
        print(f"🎙️ File giọng mẫu: {voice_path}")
        print(f"💾 File đã lưu: {output_path}")
        
        # Mở file (Windows)
        try:
            os.system(f'start "" "{output_path}"')
        except:
            print("Không thể mở file tự động. Vui lòng mở thủ công.")
    else:
        print("❌ Không thể tạo file speech")

if __name__ == "__main__":
    main()
