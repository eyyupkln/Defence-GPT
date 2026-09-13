# tokenizer.py
from collections import defaultdict
import json
import re


class BPETokenizer:
    def __init__(self, vocab_size=4000):
        self.vocab_size = vocab_size

        # Özel tokenlar (bunlar her zaman vocab'da olacak)
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<LOG>": 2,
            "<SEP>": 3,
            "<END>": 4,
        }

        self.vocab = {}  # token → id
        self.reverse_vocab = {}  # id → token
        self.merges = []  # BPE birleştirme kuralları [(a,b) → ab]


    # ADIM 1: Corpus'u kelime frekanslarına çevir

    def _build_word_freqs(self, texts):
        """
        Her kelimeyi karakterlere böl, aralarına boşluk koy.
        Sonuna </w> ekle → kelimenin bittiğini bilsin.

        Örnek:
        "BAT" → "B A T </w>"
        """
        word_freqs = defaultdict(int)

        for text in texts:
            # Özel tokenleri atla
            text = re.sub(r'<[^>]+>', '', text)
            words = text.split()

            for word in words:
                if word:
                    # Kelimeyi karakter karakter böl
                    spaced = " ".join(list(word)) + " </w>"
                    word_freqs[spaced] += 1

        return word_freqs


    # ADIM 2: Tüm çiftlerin frekansını say

    def _get_pair_freqs(self, word_freqs):
        """
        Hangi karakter çifti en sık yan yana geliyor?

        Örnek:
        "B A T </w>" → (B,A): 5, (A,T): 3, (T,</w>): 3
        """
        pair_freqs = defaultdict(int)

        for word, freq in word_freqs.items():
            symbols = word.split()

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] += freq

        return pair_freqs


    # ADIM 3: En sık çifti birleştir

    def _merge_pair(self, pair, word_freqs):
        """
        En sık geçen çifti tüm kelimelerde birleştir.

        Örnek: pair = ("B", "A")
        "B A T </w>" → "BA T </w>"
        """
        new_word_freqs = {}

        # Birleştirilecek çift için regex
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word, freq in word_freqs.items():
            new_word = pattern.sub("".join(pair), word)
            new_word_freqs[new_word] = freq

        return new_word_freqs


    # ADIM 4: Ana eğitim döngüsü
    def train(self, texts):
        """
        BPE algoritmasını çalıştır:
        1. Karakter frekanslarını say
        2. En sık çifti bul
        3. Birleştir
        4. vocab_size'a ulaşana kadar tekrarla
        """
        print("📖 Kelime frekansları hesaplanıyor...")
        word_freqs = self._build_word_freqs(texts)

        # Başlangıç vocab'ı: tüm tekil karakterler
        vocab = set()
        for word in word_freqs:
            for symbol in word.split():
                vocab.add(symbol)

        # Özel tokenleri ekle
        all_tokens = list(self.special_tokens.keys()) + sorted(vocab)
        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}

        print(f"🔤 Başlangıç vocab boyutu: {len(self.vocab)}")
        print(f"🎯 Hedef vocab boyutu: {self.vocab_size}")
        print("⚙️  BPE birleştirmeleri başlıyor...\n")

        # BPE ana döngüsü
        while len(self.vocab) < self.vocab_size:
            pair_freqs = self._get_pair_freqs(word_freqs)

            if not pair_freqs:
                break

            # En sık çifti bul
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]

            if best_freq < 2:  # Çok nadir çiftleri atla
                break

            # Birleştir
            word_freqs = self._merge_pair(best_pair, word_freqs)

            # Yeni tokeni vocab'a ekle
            new_token = "".join(best_pair)
            self.merges.append(best_pair)
            self.vocab[new_token] = len(self.vocab)
            self.reverse_vocab[len(self.reverse_vocab)] = new_token

            if len(self.vocab) % 200 == 0:
                print(
                    f"  ✅ Vocab boyutu: {len(self.vocab)} | Son birleşme: {best_pair} → '{new_token}' (frekans: {best_freq})")

        print(f"\n🏁 Eğitim tamamlandı! Final vocab boyutu: {len(self.vocab)}")


    # ADIM 5: Encode & Decode
    def encode(self, text):
        """Metni token id listesine çevir"""
        tokens = []

        # Özel tokenleri koru
        parts = re.split(r'(<[^>]+>)', text)

        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            elif part.strip():
                for word in part.split():
                    word_tokens = self._encode_word(word)
                    tokens.extend(word_tokens)

        return tokens

    def _encode_word(self, word):
        """Tek bir kelimeyi BPE kurallarıyla encode et"""
        symbols = list(word) + ["</w>"]

        # Öğrenilen merge kurallarını sırayla uygula
        for merge_pair in self.merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and \
                        symbols[i] == merge_pair[0] and \
                        symbols[i + 1] == merge_pair[1]:
                    new_symbols.append("".join(merge_pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        # Sembolleri id'ye çevir
        ids = []
        for symbol in symbols:
            if symbol in self.vocab:
                ids.append(self.vocab[symbol])
            else:
                ids.append(self.special_tokens["<UNK>"])

        return ids

    def decode(self, token_ids):
        """Token id listesini metne çevir"""
        tokens = [self.reverse_vocab.get(id, "<UNK>") for id in token_ids]
        text = " ".join(tokens)

        # </w> → boşluk (kelime sonu işaretini temizle)
        text = text.replace(" </w>", " ").replace("</w>", "")

        return text.strip()


    # KAYDET & YÜKLE
    def save(self, path="tokenizer.json"):
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Tokenizer kaydedildi: {path}")

    @classmethod
    def load(cls, path="tokenizer.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.reverse_vocab = {int(v): k for k, v in data["vocab"].items()}
        return tokenizer