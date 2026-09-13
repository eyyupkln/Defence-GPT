<p align="center">
  <h1 align="center">🛡️ Defence-GPT</h1>
  <p align="center">
    <b>Savunma Telemetri Loglarını Doğal Dile Çeviren GPT Modeli</b>
  </p>
  <p align="center">
    <a href="#-kurulum">Kurulum</a> •
    <a href="#-kullanım">Kullanım</a> •
    <a href="#-mimari">Mimari</a> •
    <a href="#-değerlendirme">Değerlendirme</a> •
    <a href="#english">English</a>
  </p>
</p>

---

## 🇹🇷 Türkçe

### 📖 Proje Hakkında

**Defence-GPT**, savunma sistemlerinden gelen ham telemetri loglarını (batarya durumu, sıcaklık, GPS, sinyal gücü vb.) insanların anlayabileceği Türkçe açıklamalara çeviren, sıfırdan eğitilmiş bir **decoder-only Transformer** modelidir.

#### Örnek

```
📥 Girdi (Ham Log):
[BAT_CRIT] BAT_VOLT=9.2V | ADDR:0x1A2B3C | TS:14:45:22

📤 Çıktı (Doğal Dil):
⚠️ BATARYA KRİTİK: Gerilim 9.2V seviyesine düştü. Sistem acil iniş protokolünü (RTH) başlattı.
```

### 🏗️ Mimari

| Bileşen | Detay |
|---|---|
| **Model** | Decoder-only Transformer (GPT-tarzı) |
| **Parametre** | ~5.2M |
| **Embedding Boyutu** | 256 |
| **Attention Heads** | 4 |
| **Transformer Katmanları** | 4 |
| **Bağlam Penceresi** | 128 token |
| **Tokenizer** | Özel BPE (Byte Pair Encoding), 4000 vocab |
| **Aktivasyon** | GELU |
| **Normalizasyon** | Pre-LayerNorm |

### 📁 Proje Yapısı

```
Defence-GPT/
├── config.yaml          # Model ve eğitim hiperparametreleri
├── model.py             # Transformer modeli (TransformerBlock + TelemetryGPT)
├── tokenizer.py         # Özel BPE tokenizer
├── tokenizer_test.py    # Tokenizer eğitim scripti
├── dataset.py           # PyTorch Dataset sınıfı
├── data_generation.py   # Sentetik telemetri verisi üretici
├── train.py             # Eğitim döngüsü (+ scheduler, checkpoint)
├── evaluate.py          # BLEU, Pattern Match, Value Accuracy metrikleri
├── inference.py         # Çeviri (inference) scripti
├── data/
│   └── telemetry_dataset.jsonl   # 50K satırlık telemetri veri seti
├── checkpoints/
│   ├── step_500.pt ... step_7000.pt  # Ara checkpoint'ler
│   └── final.pt                      # Final model ağırlıkları
├── requirements.txt     # Python bağımlılıkları
└── .gitignore           # Git dışlama kuralları
```

### 🔧 Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/eyyupkln/Defence-GPT.git
cd Defence-GPT

# 2. Sanal ortam oluştur ve aktifleştir
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

### 🚀 Kullanım

#### 1. Veri Üretimi

50.000 satırlık sentetik telemetri veri seti üretin:

```bash
python data_generation.py
```

Bu komut `data/telemetry_dataset.jsonl` dosyasını oluşturur. Desteklenen hata senaryoları:

| Hata Kodu | Açıklama | Ağırlık |
|---|---|---|
| `SYS_OK` | Sistem normal | %40 |
| `BAT_CRIT` | Kritik batarya gerilimi | %12 |
| `TEMP_HIGH` | Aşırı işlemci sıcaklığı | %12 |
| `ALT_WARN` | Düşük irtifa uyarısı | %10 |
| `GPS_LOST` | GPS uydu kaybı | %10 |
| `COMM_FAIL` | İletişim sinyal kesintisi | %9 |
| `MOT_STALL` | Motor arızası/blokaj | %7 |

#### 2. Tokenizer Eğitimi

BPE tokenizer'ı veri seti üzerinde eğitin:

```bash
python tokenizer_test.py
```

Bu komut `tokenizer.json` dosyasını oluşturur (4000 vocab boyutunda).

#### 3. Model Eğitimi

```bash
python train.py
```

Eğitim parametreleri `config.yaml` dosyasından okunur. CosineAnnealing LR scheduler ve gradient clipping kullanılır. Checkpoint'ler her 1000 adımda kaydedilir.

#### 4. Değerlendirme

```bash
python evaluate.py
```

Son 200 örnek üzerinde 3 metrik hesaplanır:
- **BLEU Score**: Üretilen metnin referansla ne kadar örtüştüğü
- **Pattern Match**: Hata koduna özgü anahtar kelimelerin varlığı
- **Value Accuracy**: Sayısal değerlerin doğru aktarılması

#### 5. Çeviri (Inference)

```bash
python inference.py
```

Eğitilmiş modeli kullanarak ham telemetri loglarını doğal dile çevirir.

### ⚙️ Hiperparametreler

Tüm ayarlar [`config.yaml`](config.yaml) dosyasından yönetilir:

```yaml
vocab_size: 4000
embedding_dim: 256
num_heads: 4
num_layers: 4
block_size: 128
dropout: 0.1
batch_size: 32
learning_rate: 0.0003
epochs: 5
eval_interval: 200
save_interval: 1000
train_split: 0.9
checkpoint_dir: "./checkpoints"
```

### 🧠 Özel Tokenlar

| Token | ID | Görev |
|---|---|---|
| `<PAD>` | 0 | Padding (dolgu) |
| `<UNK>` | 1 | Bilinmeyen token |
| `<LOG>` | 2 | Log başlangıcı |
| `<SEP>` | 3 | Girdi-çıktı ayırıcı |
| `<END>` | 4 | Dizi sonu |

### 📊 Eğitim Formatı

Model, decoder-only (autoregressive) yapıda eğitilir. Her örnek şu formattadır:

```
<LOG> [HAM_LOG] <SEP> [AÇIKLAMA] <END>
```

Model, `<SEP>` sonrasındaki açıklamayı öğrenir. Inference sırasında `<LOG> ... <SEP>` verilir ve model otomatik olarak açıklamayı üretir.

---

<a name="english"></a>
## 🇬🇧 English

### 📖 About

**Defence-GPT** is a decoder-only Transformer model trained from scratch to translate raw defence telemetry logs (battery status, temperature, GPS, signal strength, etc.) into human-readable Turkish explanations.

#### Example

```
📥 Input (Raw Log):
[BAT_CRIT] BAT_VOLT=9.2V | ADDR:0x1A2B3C | TS:14:45:22

📤 Output (Natural Language):
⚠️ BATARYA KRİTİK: Gerilim 9.2V seviyesine düştü. Sistem acil iniş protokolünü (RTH) başlattı.
(⚠️ BATTERY CRITICAL: Voltage dropped to 9.2V. System initiated emergency landing protocol (RTH).)
```

### 🏗️ Architecture

| Component | Detail |
|---|---|
| **Model** | Decoder-only Transformer (GPT-style) |
| **Parameters** | ~5.2M |
| **Embedding Dim** | 256 |
| **Attention Heads** | 4 |
| **Transformer Layers** | 4 |
| **Context Window** | 128 tokens |
| **Tokenizer** | Custom BPE (Byte Pair Encoding), 4000 vocab |
| **Activation** | GELU |
| **Normalization** | Pre-LayerNorm |

### 📁 Project Structure

```
Defence-GPT/
├── config.yaml          # Model & training hyperparameters
├── model.py             # Transformer model (TransformerBlock + TelemetryGPT)
├── tokenizer.py         # Custom BPE tokenizer implementation
├── tokenizer_test.py    # Tokenizer training script
├── dataset.py           # PyTorch Dataset class
├── data_generation.py   # Synthetic telemetry data generator
├── train.py             # Training loop (scheduler, gradient clipping, checkpoints)
├── evaluate.py          # Evaluation metrics (BLEU, Pattern Match, Value Accuracy)
├── inference.py         # Inference / translation script
├── data/
│   └── telemetry_dataset.jsonl   # 50K telemetry dataset
├── checkpoints/
│   ├── step_500.pt ... step_7000.pt  # Intermediate checkpoints
│   └── final.pt                      # Final model weights
├── requirements.txt     # Python dependencies
└── .gitignore           # Git ignore rules
```

### 🔧 Installation

```bash
# 1. Clone the repository
git clone https://github.com/eyyupkln/Defence-GPT.git
cd Defence-GPT

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 🚀 Usage

#### 1. Data Generation

Generate a 50,000-line synthetic telemetry dataset:

```bash
python data_generation.py
```

This creates the `data/telemetry_dataset.jsonl` file. Supported error scenarios:

| Error Code | Description | Weight |
|---|---|---|
| `SYS_OK` | System normal | 40% |
| `BAT_CRIT` | Critical battery voltage | 12% |
| `TEMP_HIGH` | CPU overheating | 12% |
| `ALT_WARN` | Low altitude warning | 10% |
| `GPS_LOST` | GPS satellite loss | 10% |
| `COMM_FAIL` | Communication signal loss | 9% |
| `MOT_STALL` | Motor stall / blockage | 7% |

#### 2. Tokenizer Training

Train the BPE tokenizer on the dataset:

```bash
python tokenizer_test.py
```

This generates `tokenizer.json` (4000 vocab size).

#### 3. Model Training

```bash
python train.py
```

Training parameters are loaded from `config.yaml`. Uses CosineAnnealing LR scheduler and gradient clipping. Checkpoints are saved every 1000 steps.

#### 4. Evaluation

```bash
python evaluate.py
```

Computes 3 metrics on the last 200 samples:
- **BLEU Score**: N-gram overlap between generated and reference text
- **Pattern Match**: Presence of error-code-specific keywords
- **Value Accuracy**: Correct transfer of numerical values

#### 5. Inference

```bash
python inference.py
```

Translates raw telemetry logs into natural language using the trained model.

### ⚙️ Hyperparameters

All settings are managed via [`config.yaml`](config.yaml):

```yaml
vocab_size: 4000
embedding_dim: 256
num_heads: 4
num_layers: 4
block_size: 128
dropout: 0.1
batch_size: 32
learning_rate: 0.0003
epochs: 5
eval_interval: 200
save_interval: 1000
train_split: 0.9
checkpoint_dir: "./checkpoints"
```

### 🧠 Special Tokens

| Token | ID | Purpose |
|---|---|---|
| `<PAD>` | 0 | Padding |
| `<UNK>` | 1 | Unknown token |
| `<LOG>` | 2 | Log sequence start |
| `<SEP>` | 3 | Input-output separator |
| `<END>` | 4 | End of sequence |

### 📊 Training Format

The model is trained in a decoder-only (autoregressive) manner. Each sample follows this format:

```
<LOG> [RAW_LOG] <SEP> [EXPLANATION] <END>
```

The model learns to generate the explanation after `<SEP>`. During inference, `<LOG> ... <SEP>` is provided and the model auto-generates the explanation.

---

### 📜 License

This project is licensed under the [MIT License](LICENSE).

### 👤 Author

**Eyyüp Kılın** — [GitHub](https://github.com/eyyupkln)
