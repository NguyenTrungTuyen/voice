from TTS.api import TTS
import os
import soundfile as sf
import librosa

# Tải mô hình YourTTS
model_name = "tts_models/multilingual/multi-dataset/your_tts"
tts = TTS(model_name=model_name, progress_bar=True, gpu=False)

# Văn bản nhiều dòng
text = """2 tháng 2, Long Sĩ Đầu.

Hoàng hôn, địa phương yên tĩnh trong trấn nhỏ tên là ngõ Nê Bình, có thiếu niên gầy ốm lẻ loi hiu quạnh, lúc này hắn đang dựa theo thói quen, một tay cầm ngọn nến, một tay cầm cành đào, chiếu rọi căn phòng, vách tường, giường gỗ các chỗ, dùng cành đào gõ đánh, ý đồ mượn cái này khu đuổi rắn rết, miệng lẩm bẩm, là cách ngôn trấn nhỏ này đời đời truyền xuống: 2 tháng 2, chiếu sáng nhà, đào đánh tường, rắn rết nhân gian không chỗ nấp.

Thiếu niên họ Trần, danh Bình An, cha mẹ sớm qua đời. Trấn nhỏ đồ sứ rất nổi danh, bản triều khai quốc tới nay, đã đảm đương trọng trách "Phụng chiếu nung đồ cúng tế hiến lăng", có quan viên triều đình hàng năm trú đóng nơi đây, quản lý sự vụ. Thiếu niên không chỗ dựa, từ rất sớm đã là một diêu tượng nung sứ, khởi điểm chỉ có thể làm chút việc nặng vặt vãnh, đi theo một sư phụ nửa đường tính tình khó chịu, vất vả nhịn vài năm, vừa mới thu được một chút đường lối về nung sứ, kết quả thế sự vô thường, trấn nhỏ đột nhiên mất đi lá bùa hộ mệnh nung sứ này, mấy chục cái lò nung hình như rồng nằm quanh thân trấn nhỏ, trong một đêm toàn bộ bị quan phủ cưỡng chế đóng cửa tắt lửa.



Trần Bình An buông cành đào mới bẻ, thổi tắt ngọn nến, đi ra khỏi phòng, ngồi ở bậc thềm, ngửa đầu nhìn, tinh không lấp lánh.



Thiếu niên đến nay vẫn nhớ rõ ràng, lão sư phụ chỉ chịu nhận mình là nửa đồ đệ kia, họ Diêu, ở sáng sớm tàn thu năm trước, bị người phát hiện ngồi ở trên một cái ghế trúc nhỏ, hướng đầu về phía lò nung, nhắm mắt.

Nhưng người để tâm vào chuyện vụn vặt như một người thợ già như vậy, chung quy là số ít.


Trấn nhỏ thợ thủ công đời đời chỉ biết nung sứ, vừa không dám đi quá giới hạn đi nung hàng cống phẩm, lại không dám mang đồ sứ cất trong kho ra buôn bán với dân chúng, chỉ phải đều tìm đường ra khác, Trần Bình An mười bốn tuổi cũng bị đuổi ra khỏi nhà, sau khi trở lại ngõ Nê Bình, tiếp tục thủ cái nhà cũ sớm rách nát không chịu nổi này, cảnh tượng không sai biệt lắm là chỉ có bốn bức tường ảm đạm, đó là Trần Bình An muốn làm bại gia tử, cũng không muốn ở.

Làm một đoạn thời gian cô hồn dã quỷ bay tới bay đi, thiếu niên thật sự tìm không được nghề nghiệp để kiếm tiền, dựa vào về chút tích góp ít ỏi, thiếu niên miễn cưỡng lấp đầy bụng, mấy ngày hôm trước nghe nói ở ngõ kỵ long ngoài phố, đến một lão thợ rèn họ Nguyễn vùng người, đối ngoại tuyên bố muốn thu bảy tám học đồ gõ sắt, không cho tiền công, nhưng quản cơm, Trần Bình An đã nhanh chạy tới tìm vận khí, chưa từng nghĩ lão nhân chỉ liếc mắt nhìn hắn, đã mang hắn cự ở ngoài cửa, lúc ấy Trần Bình An đã rất buồn, chẳng lẽ cái chuyện gõ sắt này, không phải xem lực cánh tay lớn nhỏ, mà là xem tướng mạo tốt xấu sao?



Phải biết rằng Trần Bình An tuy nhìn gầy yếu, nhưng khí lực không thể khinh thường, đây là thiếu niên từ nhỏ đã được rèn luyện trụ cột thân thể, trừ cái đó ra, Trần Bình An còn đi theo họ Diêu lão nhân, chạy khắp núi núi sông sông phạm vi trăm dặm khắp trấn nhỏ, biết tư vị các loại thổ nhưỡng bốn phía, chịu mệt nhọc, cái gì sống bẩn sống mệt đều nguyện ý làm, không chút nào chần chờ. Đáng tiếc lão Diêu thủy chung không thích Trần Bình An, ghét bỏ thiếu niên không có ngộ tính, là gỗ tạp không khai khiếu, xa xa không bằng đại đồ đệ Lưu Tiện Dương, cái này cũng trách không được lão nhân bất công, sư phụ đưa vào cửa, tu hành ở cá nhân, ví dụ như cùng là một cái chén đơn giản, Lưu Tiện Dương ngắn ngủn nửa năm công lực, đã ngang với tiêu chuẩn Trần Bình An vất vả ba năm.



Tuy đời này cũng chưa chắc đã cần tới cái tay nghề này nữa, nhưng Trần Bình An vẫn giống như dĩ vãng, nhắm mắt lại, tưởng tượng trước người mình lại có bàn đá cùng bánh xe, bắt đầu luyện tập làm chén, quen tay hay việc.



Đại khái qua mỗi một khắc, thiếu niên sẽ tạm nghỉ một chút, lắc lắc cổ tay, tuần hoàn lặp lại như thế, thẳng đến cả người hoàn toàn tinh bì lực tẫn, Trần Bình An lúc này mới đứng dậy, vừa tản bộ ở trong viện, vừa chậm rãi giãn ra gân cốt. Cho tới bây giờ không có ai dạy Trần Bình An cái này, là chính hắn tự tìm ra môn đạo.



Trong thiên địa nguyên bản vạn vật yên tĩnh, Trần Bình An nghe được một tiếng cười châm chọc chói tai, dừng bước chân lại, quả nhiên, nhìn thấy bạn cùng lứa tuổi ngồi xổm trên đầu tường, nhếch miệng, không chút nào che dấu thần sắc khinh rẻ của hắn.

Người này là hàng xóm cũ của Trần Bình An, nghe nói là con tư sinh Giam tạo đại nhân tiền nhiệm, vị đại nhân nọ e sợ bị thanh lưu cười chê, ngôn quan buộc tội, cuối cùng độc thân trở lại kinh thành báo cáo công tác, mang đứa nhỏ giao cho quan viên tiếp nhận chức vụ rất có quan hệ tình nghĩa cá nhân, giúp đỡ trông coi. Nay trấn nhỏ đã mất đi tư cách làm đồ sứ cho triều đình một cách khó hiểu, đốc tạo đại nhân phụ trách thay triều đình quản lý ở nơi này, chính mình cũng là Bồ Tát bùn qua sông bản thân khó bảo toàn, nào còn lo lắng con tư sinh đồng nghiệp quan trường, để lại một ít tiền mà cấp tốc chạy về kinh thành đả thông quan hệ.

Thiếu niên hàng xóm bất tri bất giác đã trở thành thứ bị vứt bỏ, qua ngày thật ra vẫn khá thoải mái, cả ngày dẫn theo nha hoàn bên người, dạo chơi ở trong ngoài trấn nhỏ, quanh năm suốt tháng chơi bời lêu lổng, cũng chưa bao giờ từng vì tóc bạc mà quá sầu.

Ngõ Nê Bình nhà nhà tường viện đất vàng đều rất thấp, thật ra thiếu niên hàng xóm không cần kiễng gót chân cũng có thể nhìn thấy cảnh tượng sân bên này, nhưng mỗi lần cùng Trần Bình An nói chuyện, cố tình thích ngồi xổm ở đầu tường.

So sánh với Trần Bình An thô thiển tục khí, thì thiếu niên hàng xóm lịch sự tao nhã hơn rất nhiều, kêu Tống Tập Tân, ngay cả tỳ nữ cùng hắn sống nương tựa lẫn nhau, cũng xưng hô có vẻ nho nhã, Trĩ Khuê.

Cô gái lúc này đứng ở bên kia tường viện, nàng có một đôi mắt hạnh, rụ rè sợ hãi.

Bên kia cửa viện, có tiếng nói vang lên, "Tỳ nữ này của ngươi có bán hay không?"

Tống Tập Tân ngẩn người, theo thanh âm quay đầu nhìn lại, là một thiếu niên cẩm y mặt mày mỉm cười, đứng ở ngoài viện, một gương mặt hoàn toàn xa lạ.

Bên cạnh thiếu niên cẩm y đứng một vị lão giả thân hình cao lớn, khuôn mặt trắng nõn, sắc mặt hòa ái, nhẹ nhàng hí mắt đánh giá thiếu nhiên thiếu nữ trong hai tòa nhà giáp sân.

Tầm mắt lão giả đảo qua Trần Bình An, cũng không đình trệ, nhưng mà ở trên người Tống Tập Tân cùng tỳ nữ, hơi có dừng lại, ý cười dần dần nồng đậm.

Tống Tập Tân liếc mắt nói: "Bán! Sao lại không bán!"

Thiếu niên nọ mỉm cười nói: "Vậy ngươi nói cái giá."


Cô gái trừng lớn đôi mắt, vẻ mặt không thể tưởng tượng, giống một con nai con.

Tống Tập Tân liếc cái xem thường, vươn một ngón tay, lắc lắc, "Bạc trắng một vạn lượng!"

Thiếu niên cẩm y sắc mặt như thường, gật đầu nói: "Tốt."

Tống Tập Tân thấy thiếu niên nọ bộ dáng không giống như là nói đùa, vội vàng sửa lời nói: "Là hoàng kim vạn lượng!"

Thiếu niên cẩm y khóe miệng nhếch lên, nói: "Chọc ngươi thôi."

Tống Tập Tân sắc mặt âm trầm."""

# Lấy đường dẫn tuyệt đối
base_dir = os.path.dirname(os.path.abspath(__file__))
voice_path = os.path.join(base_dir, "dyi.wav")
output_path = os.path.join(base_dir, "output.wav")

# --- Kiểm tra file dyi.wav ---
def check_wav_file(file_path):
    try:
        # Đọc file WAV
        y, sr = librosa.load(file_path, sr=None)
        # Kiểm tra định dạng
        if not file_path.endswith('.wav'):
            print("Lỗi: File phải có định dạng WAV.")
            return False
        if len(y.shape) > 1:
            print("Lỗi: File WAV phải là mono, không phải stereo.")
            return False
        if sr not in [16000, 22050, 44100]:
            print(f"Lỗi: Tần số mẫu {sr}Hz không được hỗ trợ. Chuyển về 16000Hz hoặc 22050Hz.")
            return False
        return True
    except Exception as e:
        print(f"Lỗi khi đọc file dyi.wav: {e}")
        return False

# --- Xử lý TTS ---
try:
    if os.path.exists(voice_path) and check_wav_file(voice_path):
        print(f"Đang clone giọng từ: {voice_path}")
        # Clone giọng từ dyi.wav, thêm language="en" để tránh lỗi
        tts.tts_to_file(text=text, speaker_wav=voice_path, language="en", file_path=output_path)
    else:
        print("File dyi.wav không hợp lệ hoặc không tồn tại, sử dụng giọng mặc định (tiếng Anh).")
        tts.tts_to_file(text=text, language="en", file_path=output_path)

    # In đường dẫn để kiểm tra
    print(f"🎙️ Voice mẫu: {voice_path}")
    print(f"💾 Đã lưu file: {output_path}")

    # Phát bằng trình nghe nhạc mặc định (Windows)
    os.system(f'start "" "{output_path}"')
except Exception as e:
    print(f"Lỗi: {e}")