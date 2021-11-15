import main
import torch
import torch.nn.functional as F

m = main.Model.load_from_checkpoint('epoch=825-step=13215.ckpt')
m.eval()
while True:
    expr = input("Prompt: ")
    curi = 0
    past = None
    max_length = 64
    print(expr, end="", flush=True)
    with torch.no_grad():
        while len(expr) < max_length:
            #model generates solution character by character, recursively feeding through model
            cur = torch.tensor([bytearray(expr[curi:], 'ascii')], dtype=torch.long)
            probs, past = m.model(cur, past=past)
            curi = len(expr)
            probs = F.softmax(probs[0, -1], dim=-1)
            sample = torch.multinomial(probs, 1)[0]
            expr += chr(sample)
            print(chr(sample), end="", flush=True)
            if sample == ord(';'):
                print("")
                break
