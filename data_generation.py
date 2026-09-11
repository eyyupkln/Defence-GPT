import json
import random
from datetime import datetime


class TelemetryGenerator:
    def __init__(self):
        # 1. Hata kodları, ilgili sensörler ve değer aralıkları (Gerçekçi korelasyon)
        self.scenarios = {
            "BAT_CRIT": {
                "sensor": "BAT_VOLT", "unit": "V",
                "val_range": (8.5, 10.4), "round_digits": 2,
                "template": lambda
                    v: f"⚠️ BATARYA KRİTİK: Gerilim {v}V seviyesine düştü. Sistem acil iniş protokolünü (RTH) başlattı."
            },
            "TEMP_HIGH": {
                "sensor": "TEMP_CPU", "unit": "°C",
                "val_range": (81.0, 98.0), "round_digits": 1,
                "template": lambda
                    v: f"🔥 ISI ALARMI: İşlemci sıcaklığı {v}°C. Aşırı ısınma nedeniyle soğutma fanları %100 güce alındı."
            },
            "ALT_WARN": {
                "sensor": "ALT", "unit": "M",
                "val_range": (10, 85), "round_digits": 0,
                "template": lambda
                    v: f"📍 ALÇAK İRTİFA UYARISI: İrtifa {int(v)}m. Araç kritik irtifa eşiğinin altında seyrediyor."
            },
            "GPS_LOST": {
                "sensor": "GPS_SAT", "unit": "cnt",
                "val_range": (0, 2), "round_digits": 0,
                "template": lambda
                    v: f"❌ GPS KAYBI: Kilitli uydu sayısı {int(v)}. Ataletsel Seyrüsefer Sistemi (INS) aktif edildi."
            },
            "COMM_FAIL": {
                "sensor": "RSSI", "unit": "dBm",
                "val_range": (-120, -105), "round_digits": 0,
                "template": lambda
                    v: f"📡 SİNYAL KESİNTİSİ: RSSI {int(v)}dBm. Yer istasyonu bağlantısı koptu, otonom seyir devrede."
            },
            "MOT_STALL": {
                "sensor": "SPEED", "unit": "M/S",
                "val_range": (0.0, 2.5), "round_digits": 2,
                "template": lambda
                    v: f"⚡ MOTOR ARIZASI: İlerleme hızı {v}m/s seviyesine geriledi. Motor blokajı şüphesi mevcut."
            },
            "SYS_OK": {
                "sensor": "BAT_VOLT", "unit": "V",
                "val_range": (11.5, 12.6), "round_digits": 2,
                "template": lambda
                    v: f"✅ SİSTEM NORMAL: Batarya {v}V seviyesinde ve telemetri değerleri nominal aralıkta."
            }
        }
        self.scenario_weights = {
            "SYS_OK": 0.40,  # %40 normal
            "BAT_CRIT": 0.12,
            "TEMP_HIGH": 0.12,
            "ALT_WARN": 0.10,
            "GPS_LOST": 0.10,
            "COMM_FAIL": 0.09,
            "MOT_STALL": 0.07,
        }

    def generate_log_entry(self):
        # Rastgele bir senaryo seç
        error_code = random.choice(list(self.scenarios.keys()))
        config = self.scenarios[error_code]

        # Mantıklı aralıkta sensör değeri üret
        low, high = config["val_range"]
        val = random.uniform(low, high)
        if config["round_digits"] == 0:
            val = int(val)
        else:
            val = round(val, config["round_digits"])

        # Hexadecimal donanım adresi ve Timestamp
        hex_code = f"0x{random.randint(0, 0xFFFFFF):06X}"
        time_str = datetime.now().strftime("%H:%M:%S")

        # Ham telemetri logu
        raw_log = f"[{error_code}] {config['sensor']}={val}{config['unit']} | ADDR:{hex_code} | TS:{time_str}"
        explanation = config["template"](val)

        # Decoder (GPT) Eğitimi İçin Birleşik Metin:
        # GPT modelleri bir diziyi baştan sona tahmin ettiği için
        # girdiyi ve hedefi özel bir ayraç ile tek bir metin yaparız.
        combined_text = f"<LOG> {raw_log} <SEP> {explanation} <END>"

        error_code = random.choices(
            list(self.scenarios.keys()),
            weights=list(self.scenario_weights.values())
        )[0]

        return {
            "input": raw_log,
            "target": explanation,
            "full_text": combined_text
        }


# --- Veri Üretim Süreci ---
if __name__ == "__main__":
    generator = TelemetryGenerator()
    dataset_size = 50000

    print(f"{dataset_size} satırlık telemetri veri seti üretiliyor...")

    with open("data/telemetry_dataset.jsonl", "w", encoding="utf-8") as f:
        for _ in range(dataset_size):
            entry = generator.generate_log_entry()
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("✅ telemetry_dataset.jsonl dosyası başarıyla oluşturuldu!")

    # İlk üretilen örneği ekrana basıp kontrol edelim
    sample = generator.generate_log_entry()
    print("\n--- Örnek Veri ---")
    print(f"Formatlı Tam Metin: {sample['full_text']}")