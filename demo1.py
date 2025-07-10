from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
import os

model_name = "facebook/mms-tts-vie"  # Mô hình giả định
try:
    model = VitsModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = """
2 tháng 2, Long Sĩ Đầu.

Hoàng hôn, địa phương yên tĩnh trong trấn nhỏ tên là ngõ Nê Bình, có thiếu niên gầy ốm lẻ loi hiu quạnh, lúc này hắn đang dựa theo thói quen, một tay cầm ngọn nến, một tay cầm cành đào, chiếu rọi căn phòng, vách tường, giường gỗ các chỗ, dùng cành đào gõ đánh, ý đồ mượn cái này khu đuổi rắn rết, miệng lẩm bẩm, là cách ngôn trấn nhỏ này đời đời truyền xuống: 2 tháng 2, chiếu sáng nhà, đào đánh tường, rắn rết nhân gian không chỗ nấp.

Thiếu niên họ Trần, danh Bình An, cha mẹ sớm qua đời. Trấn nhỏ đồ sứ rất nổi danh, bản triều khai quốc tới nay, đã đảm đương trọng trách "Phụng chiếu nung đồ cúng tế hiến lăng", có quan viên triều đình hàng năm trú đóng nơi đây, quản lý sự vụ. Thiếu niên không chỗ dựa, từ rất sớm đã là một diêu tượng nung sứ, khởi điểm chỉ có thể làm chút việc nặng vặt vãnh, đi theo một sư phụ nửa đường tính tình khó chịu, vất vả nhịn vài năm, vừa mới thu được một chút đường lối về nung sứ, kết quả thế sự vô thường, trấn nhỏ đột nhiên mất đi lá bùa hộ mệnh nung sứ này, mấy chục cái lò nung hình như rồng nằm quanh thân trấn nhỏ, trong một đêm toàn bộ bị quan phủ cưỡng chế đóng cửa tắt lửa.



Trần Bình An buông cành đào mới bẻ, thổi tắt ngọn nến, đi ra khỏi phòng, ngồi ở bậc thềm, ngửa đầu nhìn, tinh không lấp lánh.



Thiếu niên đến nay vẫn nhớ rõ ràng, lão sư phụ chỉ chịu nhận mình là nửa đồ đệ kia, họ Diêu, ở sáng sớm tàn thu năm trước, bị người phát hiện ngồi ở trên một cái ghế trúc nhỏ, hướng đầu về phía lò nung, nhắm mắt.

Nhưng người để tâm vào chuyện vụn vặt như một người thợ già như vậy, chung quy là số ít.


Trấn nhỏ thợ thủ công đời đời chỉ biết nung sứ, vừa không dám đi quá giới hạn đi nung hàng cống phẩm, lại không dám mang đồ sứ cất trong kho ra buôn bán với dân chúng, chỉ phải đều tìm đường ra khác, Trần Bình An mười bốn tuổi cũng bị đuổi ra khỏi nhà, sau khi trở lại ngõ Nê Bình, tiếp tục thủ cái nhà cũ sớm rách nát không chịu nổi này, cảnh tượng không sai biệt lắm là chỉ có bốn bức tường ảm đạm, đó là Trần Bình An muốn làm bại gia tử, cũng không muốn ở.

Làm một đoạn thời gian cô hồn dã quỷ bay tới bay đi, thiếu niên thật sự tìm không được nghề nghiệp để kiếm tiền, dựa vào về chút tích góp ít ỏi, thiếu niên miễn cưỡng lấp đầy bụng, mấy ngày hôm trước nghe nói ở ngõ kỵ long ngoài phố, đến một lão thợ rèn họ Nguyễn vùng người, đối ngoại tuyên bố muốn thu bảy tám học đồ gõ sắt, không cho tiền công, nhưng quản cơm, Trần Bình An đã nhanh chạy tới tìm vận khí, chưa từng nghĩ lão nhân chỉ liếc mắt nhìn hắn, đã mang hắn cự ở ngoài cửa, lúc ấy Trần Bình An đã rất buồn, chẳng lẽ cái chuyện gõ sắt này, không phải xem lực cánh tay lớn nhỏ, mà là xem tướng mạo tốt xấu sao?

"""

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():  # Sửa cú pháp từ with_with thành with
        output = model(**inputs).waveform

    # Lưu file âm thanh
    sf.write("output.wav", output.squeeze().numpy(), model.config.sampling_rate)
    print("💾 Đã lưu file: output.wav")
    os.system(f'start "" "output.wav"')
except Exception as e:
    print(f"Lỗi: {e}")