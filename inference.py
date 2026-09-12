import torch
from tokenizer import BPETokenizer
from model import TelemetryGPT



def load_model(checkpoint_path, device):
    tokenizer=BPETokenizer.load("tokenizer.json")

    model=TelemetryGPT(
        vocab_size=2000,
        embedding_dim=256,
        num_heads=4,
        num_layers=4,
        block_size=128
    ).to(device)


    checkpoint=torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["Model"])
    model.eval()

    return model,tokenizer

def translate(model , tokenizer,raw_log,device ,max_new_tokens=80,temperature=1):
    # Girdiyi formatla
    prompt = f"<LOG> {raw_log} <SEP>"

    input_ids=tokenizer.encode(prompt)
    input_tensor=torch.tensor([input_ids],dtype=torch.long).to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(input_tensor, max_new_tokens=max_new_tokens, temperature=temperature)

    # Sadece yeni üretilen kısmı decode et
    new_tokens=output_ids[0][len(input_ids):].tolist()
    result=tokenizer.decode(new_tokens)

    # <END>'e kadar kes
    if "<END>" in result:
        result = result.split("<END>")[0].strip()

    return result


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer=load_model(checkpoint_path="./checkpoints/final.pt",device=device)
    test_logs = [
        "[BAT_CRIT] BAT_VOLT=9.2V | ADDR:0x1A2B3C | TS:14:45:22",
        "[TEMP_HIGH] TEMP_CPU=92.5°C | ADDR:0xDEADBF | TS:15:12:01",
        "[GPS_LOST] GPS_SAT=1cnt | ADDR:0x123456 | TS:15:30:45",
        "[SYS_OK] BAT_VOLT=12.1V | ADDR:0xABCDEF | TS:16:00:00",
    ]

    print("=" * 60)
    for log in test_logs:
        result = translate(model, tokenizer, log, device)
        print(f"📥 {log}")
        print(f"📤 {result}")
        print("-" * 60)












