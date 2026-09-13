import torch
import json
from tokenizer import BPETokenizer
from model import TelemetryGPT
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re



def load_model(checkpoint_path , device ):
    tokenizer = BPETokenizer.load("tokenizer.json")
    model = TelemetryGPT(
        vocab_size= 4000,
        embedding_dim = 256,
        num_heads = 4,
        num_layers  = 4,
        block_size = 128
    ).to(device)

    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["Model"])
    model.eval()
    return model , tokenizer



def translate ( model , tokenizer , raw_log , device , max_new_tokens=80):
    prompt = f"<LOG> {raw_log} <SEP>"
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output_ids = model.generate(input_tensor, max_new_tokens=max_new_tokens)

    new_tokens = output_ids[0][len(input_ids):].tolist()
    result = tokenizer.decode(new_tokens)

    if "<END>" in result:
        result = result.split("<END>")[0].strip()

    return result


def compute_bleu(reference ,hypothesis ):
    """
        reference  : Gerçek hedef metin
        hypothesis : Modelin ürettiği metin
    """

    ref_tokens= reference.split()
    hyp_tokens=hypothesis.split()
    smoothie = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    return score


ERROR_KEYWORDS = {
    "BAT_CRIT" : ["BATARYA", "KRİTİK", "RTH"],
    "TEMP_HIGH": ["ISI", "ALARMI", "soğutma"],
    "ALT_WARN" : ["İRTİFA", "UYARISI"],
    "GPS_LOST" : ["GPS", "KAYBI", "INS"],
    "COMM_FAIL": ["SİNYAL", "KESİNTİSİ", "otonom"],
    "MOT_STALL": ["MOTOR", "ARIZASI", "blokaj"],
    "SYS_OK"   : ["SİSTEM", "NORMAL", "nominal"],
}


def compute_pattern_match(error_code, hypothesis):
    """
    Üretilen metinde beklenen anahtar kelimeler var mı?
    """
    if error_code not in ERROR_KEYWORDS:
        return 0.0

    keywords = ERROR_KEYWORDS[error_code]
    matches  = sum(1 for kw in keywords if kw in hypothesis)
    return matches / len(keywords)



def compute_value_accuracy(raw_log, hypothesis):
    """
    Girdideki sayısal değer çıktıda doğru geçiyor mu?
    Örnek: BAT_VOLT=9.2V → çıktıda "9.2" var mı?
    """
    # Girdiden sayıyı çek (= ile birim arasındaki kısım)
    match = re.search(r"=([0-9.]+)", raw_log)
    if not match:
        return None

    value = match.group(1)
    return 1.0 if value in hypothesis else 0.0



def evaluate(checkpoint_path,jsonl_path,num_samples=200):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print (f"Device:{device}")

    model, tokenizer = load_model(checkpoint_path, device)

    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    test_data = data[-num_samples:]

    bleu_scores = []
    pattern_scores = []
    value_scores = []

    print(f"\n{num_samples} örnek üzerinde değerlendirme yapılıyor...\n")

    for i , entry in enumerate(test_data):
        raw_log=entry["input"]
        reference=entry["target"]
        error_code = raw_log.split("]")[0].replace("[", "").strip()

        # Çeviri yap
        hypothesis = translate(model, tokenizer, raw_log, device)

        # Metrikler
        bleu = compute_bleu(reference, hypothesis)
        pattern = compute_pattern_match(error_code, hypothesis)
        value = compute_value_accuracy(raw_log, hypothesis)

        bleu_scores.append(bleu)
        pattern_scores.append(pattern)
        if value is not None:
            value_scores.append(value)

        # İlk 5 örneği ekrana bas
        if i < 5:
            print(f"Input     : {raw_log}")
            print(f"Reference : {reference}")
            print(f"Hypothesis: {hypothesis}")
            print(f"   BLEU: {bleu:.3f} | Pattern: {pattern:.2f} | Value: {value}")
            print("-" * 60)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_pattern = sum(pattern_scores) / len(pattern_scores)
    avg_value = sum(value_scores) / len(value_scores) if value_scores else 0.0

    print("\n" + "=" * 60)
    print("DEĞERLENDİRME SONUÇLARI\n")
    print(f"  BLEU Score      : {avg_bleu:.4f}  (1.0 = mükemmel)")
    print(f"  Pattern Match   : {avg_pattern:.4f}  (1.0 = tüm anahtar kelimeler var)")
    print(f"  Value Accuracy  : {avg_value:.4f}  (1.0 = sayısal değer doğru)")

    return {
        "bleu": avg_bleu,
        "pattern": avg_pattern,
        "value": avg_value
    }



if __name__ == "__main__":
    evaluate(
        checkpoint_path="./checkpoints/final.pt",
        jsonl_path="data/telemetry_dataset.jsonl",
        num_samples=200
    )


