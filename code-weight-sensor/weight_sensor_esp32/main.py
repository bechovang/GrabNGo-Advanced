# ================== CÁC THƯ VIỆN CẦN THIẾT ==================
from machine import Pin
from hx711 import HX711
import time
import network
from umqtt.simple import MQTTClient

# ================== CẤU HÌNH MẠNG VÀ MQTT ==================
WIFI_SSID = "Hshop Guest"
WIFI_PASSWORD = "dienturobot"

MQTT_BROKER = "test.mosquitto.org" # Dùng broker công cộng để test
MQTT_CLIENT_ID = "esp32-shelf-1"   # Đặt tên riêng cho thiết bị của bạn
MQTT_TOPIC = "my-shop/shelf-1/events" # Chủ đề để gửi dữ liệu

# ================== CẤU HÌNH CHÂN ==================
DT_PIN = 25   # ESP32 GPIO 25 → HX711 DT (DOUT)
SCK_PIN = 26  # ESP32 GPIO 26 → HX711 SCK (PD_SCK)

# ================== GIÁ TRỊ HIỆU CHUẨN (Cập nhật theo đo của bạn) ==================
TARE_VALUE = 471778
VALUE_WITH_WEIGHT = 256326
KNOWN_WEIGHT_G = 480
# Có thể để công thức hoặc dùng giá trị số trực tiếp:
RATIO = (VALUE_WITH_WEIGHT - TARE_VALUE) / KNOWN_WEIGHT_G
#RATIO = -452.4

# ================== KHỞI TẠO CẢM BIẾN ==================
print("[START] Khoi dong can...")
print(f"[CFG] Cau hinh chan: DT={DT_PIN}, SCK={SCK_PIN}")
hx = HX711(d_out=DT_PIN, pd_sck=SCK_PIN)
time.sleep(1)

# Test đọc HX711 ngay sau khi khởi tạo
print("[TEST] Dang test doc HX711...")
test_readings = []
for i in range(5):
    try:
        val = hx.read()
        test_readings.append(val)
        print(f"   Lần {i+1}: {val}")
    except Exception as e:
        print(f"   [ERROR] Loi doc lan {i+1}: {e}")
    time.sleep(0.1)

if all(r == 0 for r in test_readings):
    print("[WARN] CANH BAO: Tat ca gia tri doc deu = 0!")
    print("[TIP] Kiem tra:")
    print("   1. Dây kết nối DT (GPIO {}) và SCK (GPIO {})".format(DT_PIN, SCK_PIN))
    print("   2. Load cell có kết nối đúng với HX711 không")
    print("   3. HX711 có được cấp nguồn (VCC/GND) không")
    print("   4. Thử đổi chân DT/SCK nếu cần")
else:
    print(f"[OK] HX711 dang doc duoc gia tri (trung binh: {sum(test_readings)/len(test_readings):.0f})")
print()

# ================== KẾT NỐI WIFI ==================
print("[WIFI] Dang khoi tao Wi-Fi...")
wlan = network.WLAN(network.STA_IF)
wlan.active(False)  # Tắt trước để reset
time.sleep(0.5)
wlan.active(True)   # Bật lại
time.sleep(1)       # Đợi Wi-Fi sẵn sàng

# Quét mạng Wi-Fi để kiểm tra SSID có sẵn không
print("[SCAN] Dang quet mang Wi-Fi...")
try:
    networks = wlan.scan()
    print(f"[NET] Tim thay {len(networks)} mang Wi-Fi:")
    found_ssid = False
    for net in networks:
        ssid = net[0].decode('utf-8') if isinstance(net[0], bytes) else net[0]
        rssi = net[3]  # Signal strength
        print(f"   - {ssid} (Signal: {rssi} dBm)")
        if ssid == WIFI_SSID:
            found_ssid = True
            print(f"   [OK] Tim thay mang '{WIFI_SSID}'!")
    
    if not found_ssid:
        print(f"[WARN] CANH BAO: Khong tim thay mang '{WIFI_SSID}' trong danh sach!")
        print("[TIP] Kiem tra lai ten mang (SSID) co dung khong, hoac mang co the bi an.")
    else:
        print(f"[OK] Mang '{WIFI_SSID}' co san, dang thu ket noi...")
except Exception as e:
    print(f"[WARN] Khong the quet mang: {e}")
    print("[TIP] Tiep tuc thu ket noi...")

if not wlan.isconnected():
    print(f"[WIFI] Dang ket noi toi Wi-Fi: {WIFI_SSID}...")
    try:
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    except OSError as e:
        print(f"[ERROR] Loi ket noi: {e}")
        print("[RETRY] Dang thu lai...")
        wlan.active(False)
        time.sleep(1)
        wlan.active(True)
        time.sleep(1)
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    
    # Chờ kết nối với timeout
    max_wait = 20
    while not wlan.isconnected() and max_wait > 0:
        time.sleep(1)
        max_wait -= 1
        if max_wait % 5 == 0:
            print(f"[WAIT] Dang cho ket noi... ({max_wait}s)")
    
    if wlan.isconnected():
        print(f"[OK] Da ket noi Wi-Fi! IP: {wlan.ifconfig()[0]}")
    else:
        print("[ERROR] Khong the ket noi Wi-Fi sau 20 giay!")
        print("[TIP] Kiem tra lai SSID va mat khau, hoac khoang cach toi router.")
        raise Exception("Wi-Fi connection failed")
else:
    print(f"[OK] Da ket noi Wi-Fi! IP: {wlan.ifconfig()[0]}")

# ================== KẾT NỐI MQTT BROKER ==================
print(f"[MQTT] Dang ket noi toi MQTT Broker: {MQTT_BROKER}...")
client = MQTTClient(MQTT_CLIENT_ID, MQTT_BROKER)
client.connect()
print("[OK] Da ket noi MQTT Broker!")

# ================== CÁC HÀM XỬ LÝ (Giữ nguyên) ==================
def read_weight_stable(samples=10):
    readings = []
    # Bỏ qua vài lần đọc đầu tiên có thể không ổn định
    for _ in range(3):
        try:
            hx.read()
        except:
            pass
        time.sleep_ms(10)
        
    for _ in range(samples):
        try:
            val = hx.read()
            readings.append(val)
        except Exception as e:
            # Nếu lỗi, thêm 0 hoặc giá trị cuối cùng
            if readings:
                readings.append(readings[-1])
            else:
                readings.append(0)
        time.sleep_ms(10)
    
    if not readings or all(r == 0 for r in readings):
        return 0
    
    return sorted(readings)[len(readings) // 2]

def convert_to_weight(reading):
    return (reading - TARE_VALUE) / RATIO

# ================== VÒNG LẶP CHÍNH ĐÃ NÂNG CẤP ==================
last_known_weight = 0
WEIGHT_CHANGE_THRESHOLD = 50  # Chỉ gửi tín hiệu nếu trọng lượng thay đổi > 50g

# Đọc khối lượng ban đầu để làm mốc so sánh
initial_raw = read_weight_stable()
last_known_weight = convert_to_weight(initial_raw)
print(f"[WEIGHT] Khoi luong ban dau on dinh: {last_known_weight:.1f} g")
print("[LOOP] Bat dau vong lap doc can...")
print("[TIP] He thong dang chay. Them/bot vat tren can de test MQTT.\n")

# Biến để hiển thị heartbeat
loop_count = 0
last_heartbeat_time = time.time()

while True:
    try:
        raw = read_weight_stable()
        current_weight = convert_to_weight(raw)
        loop_count += 1
        
        weight_change = current_weight - last_known_weight
        
        # Hiển thị heartbeat mỗi 5 giây để biết code vẫn chạy
        current_time = time.time()
        if current_time - last_heartbeat_time >= 5:
            print(f"[HEART] Dang chay... (Lan doc: {loop_count})")
            print(f"   [DATA] Raw HX711: {raw}")
            if raw == 0:
                print(f"   [WARN] CANH BAO: Raw = 0! HX711 khong doc duoc gia tri!")
                print(f"   [TIP] Kiem tra ket noi day DT (GPIO {DT_PIN}) va SCK (GPIO {SCK_PIN})")
            print(f"   [WEIGHT] Khoi luong: {current_weight:.1f} g")
            print(f"   [CHANGE] Thay doi so voi moc: {weight_change:.1f} g")
            print(f"   [THRESH] Nguong: ±{WEIGHT_CHANGE_THRESHOLD} g\n")
            last_heartbeat_time = current_time
        
        # KIỂM TRA SỰ THAY ĐỔI ĐÁNG KỂ
        if abs(weight_change) > WEIGHT_CHANGE_THRESHOLD:
            # Làm tròn giá trị thay đổi
            change_to_report = round(weight_change)
            
            print(f"[ALERT] Phat hien thay doi: {change_to_report} g. Dang gui tin hieu...")
            
            # Tạo payload và gửi qua MQTT
            payload = f"CHANGE:{change_to_report}"
            client.publish(MQTT_TOPIC, payload)
            
            print(f"[OK] Da gui: '{payload}' toi topic '{MQTT_TOPIC}'")
            
            # Cập nhật lại khối lượng đã biết để so sánh cho lần sau
            last_known_weight = current_weight
            
            # Chờ một chút để tránh gửi liên tục
            time.sleep(2) 
            
    except Exception as e:
        print(f"Lỗi: {e}. Đang thử kết nối lại...")
        # Nếu có lỗi (mất kết nối...), thử kết nối lại
        time.sleep(5)
        try:
            client.connect()
        except:
            print("Kết nối lại thất bại.")

    time.sleep(0.2) # Giảm tần suất đọc để hệ thống ổn định

