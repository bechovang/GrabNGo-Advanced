# ================== TEST CẢM BIẾN TRỌNG LƯỢNG HX711 ==================
# File này dùng để test nhanh xem cân có hoạt động không
# Chạy: exec(open('test_weight.py').read())

from machine import Pin
from hx711 import HX711
import time

# CẤU HÌNH CHÂN (Đảm bảo đúng với kết nối硬件)
DT_PIN = 25   # ESP32 GPIO 25 → HX711 DT (DOUT)
SCK_PIN = 26  # ESP32 GPIO 26 → HX711 SCK (PD_SCK)

print("🚀 Bắt đầu test cân HX711...")
print(f"📌 Cấu hình chân: DT={DT_PIN}, SCK={SCK_PIN}")

# Khởi tạo cảm biến
print("🔌 Đang khởi tạo HX711...")
try:
    hx = HX711(d_out=DT_PIN, pd_sck=SCK_PIN)
    time.sleep(1)
    print("✅ Đã khởi tạo HX711 thành công!")
except Exception as e:
    print(f"❌ Lỗi khởi tạo HX711: {e}")
    print("💡 Kiểm tra kết nối chân DT và SCK")
    raise

# Test đọc giá trị thô từ cảm biến
print("\n🔍 Đọc giá trị thô (raw) từ cảm biến...")
print("   Ghi chú:")
print("   - Nếu tất cả giá trị = 0: Dây kết nối có vấn đề")
print("   - Nếu giá trị dao động lớn: Cần hiệu chuẩn lại")
print("   - Nếu giá trị ổn định: Cảm biến đang hoạt động tốt\n")

readings = []
for i in range(10):
    try:
        val = hx.read()
        readings.append(val)
        print(f"   Lần {i+1}: {val}")
        time.sleep(0.2)
    except Exception as e:
        print(f"   ❌ Lỗi đọc lần {i+1}: {e}")
        readings.append(0)
        time.sleep(0.2)

# Phân tích kết quả
if all(r == 0 for r in readings):
    print("\n⚠️  CẢNH BÁO: Tất cả giá trị đọc đều = 0!")
    print("💡 Gợi ý kiểm tra:")
    print("   1. Dây kết nối DT (GPIO 25) và SCK (GPIO 26)")
    print("   2. Load cell có kết nối đúng với HX711 không")
    print("   3. HX711 có được cấp nguồn (VCC/GND) không")
    print("   4. Thử đổi chân DT/SCK nếu cần")
elif len(set(readings)) == 1:
    print("\n⚠️  CẢNH BÁO: Tất cả giá trị đọc đều giống nhau!")
    print("💡 Có thể do:")
    print("   1. Cảm biến đang bị kẹt/mech")
    print("   2. Load cell quá tải hoặc không tải")
    print("   3. Cần reset ESP32 và thử lại")
else:
    avg_val = sum(readings) / len(readings)
    min_val = min(readings)
    max_val = max(readings)
    print(f"\n✅ Cảm biến đang hoạt động!")
    print(f"📊 Thống kê 10 lần đọc:")
    print(f"   - Trung bình: {avg_val:.0f}")
    print(f"   - Min: {min_val}")
    print(f"   - Max: {max_val}")
    print(f"   - Dao động: {max_val - min_val}")

# Test zero/tare (đặt lại giá trị 0)
print("\n🔄 Test TARE (đặt lại giá trị 0)...")
try:
    hx.tare()
    print("✅ Đã thực hiện TARE thành công!")
    
    # Đọc giá trị sau khi tare
    print("📖 Đọc giá trị sau TARE:")
    tare_readings = []
    for i in range(5):
        val = hx.read()
        tare_readings.append(val)
        print(f"   Lần {i+1}: {val}")
        time.sleep(0.2)
    
    if all(abs(r) < 1000 for r in tare_readings):
        print("✅ TARE thành công! Giá trị gần 0")
    else:
        print("⚠️  TARE có thể không thành công")
        print(f"   Giá trị trung bình sau TARE: {sum(tare_readings)/len(tare_readings):.0f}")
except Exception as e:
    print(f"❌ Lỗi khi thực hiện TARE: {e}")

print("\n🏁 Kết thúc test cân!")
print("💡 Nếu cân hoạt động tốt, bạn có thể chạy file main.py để kết nối MQTT")
print("💡 Nếu có lỗi, hãy kiểm tra lại kết nối phần cứng")