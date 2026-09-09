# test_tokenizer.py
import json

from tokenizer import BPETokenizer

# Veriyi yükle
texts = []
with open("data/telemetry_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        texts.append(entry["full_text"])

# Tokenizer'ı eğit
tokenizer = BPETokenizer(vocab_size=2000)
tokenizer.train(texts)
tokenizer.save("tokenizer.json")

# Test
test = "<LOG> [BAT_CRIT] BAT_VOLT=9.87V | ADDR:0xA3F2C1 | TS:14:32:09 <SEP>"
encoded = tokenizer.encode(test)
decoded = tokenizer.decode(encoded)

print(f"\n📥 Orijinal : {test}")
print(f"🔢 Encoded  : {encoded}")
print(f"📤 Decoded  : {decoded}")
print(f"📊 Token sayısı: {len(encoded)}")