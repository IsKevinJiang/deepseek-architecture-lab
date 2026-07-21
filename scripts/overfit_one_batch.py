import torch
import torch.nn as nn

from deepseek_lab.model import Model

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(64, 4, 16, 128, 32, 2)
model.to(device)
model.train()

token_ids = torch.randint(0, model.vocab_size, size=(4, 17), device=device)
inputs = token_ids[:, :-1]
targets = token_ids[:, 1:]

optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
criterion = nn.CrossEntropyLoss()

for step in range(501):
    optimizer.zero_grad(set_to_none=True)

    logits = model(inputs)
    loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
    loss.backward()
    optimizer.step()

    if step % 25 == 0:
        print(f"step {step:3d} | loss {loss.item():.4f}")
