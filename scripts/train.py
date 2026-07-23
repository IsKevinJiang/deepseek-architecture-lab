import torch
import torch.nn as nn 
from modelforge.model import Model
from modelforge.data import ShardDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if (not torch.cuda.is_bf16_supported() or torch.cuda.is_available()):
    raise RuntimeError("No GPU or BF16 supported")

torch.manual_seed(10)
model = Model(128, 4, 128, 512, 50257, 4)
model.to(device)

train = ShardDataLoader("data/fineweb_smoke", 4, 128, "train", 10)
val = ShardDataLoader("data/fineweb_smoke", 4, 128, "val", 10)
val_state = val.state_dict()

optimzer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

def train_step():
    model.train()
    # Takes a batch of data and sends it to GPU
    inputs, targets = train.next_batch()
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimzer.zero_grad()
    with torch.autocast(device_type= "cuda", dtype=torch.bfloat16):
        logits = model(inputs)

        #Reshaping logits and targets for loss function
        logits = logits.reshape([512,50257])
        targets = targets.flatten()
        loss = loss_fn(logits, targets)
    loss.backward()
    optimzer.step()

    return loss.item()

def evaluate(steps):
    if steps <= 0:
        raise ValueError("Steps must be postives")
    running_loss = 0
    val.load_state_dict(val_state)
    model.eval()

    with torch.no_grad():
        for _ in range(steps):
            inputs, targets = val.next_batch()
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.autocast(device_type= "cuda", dtype=torch.bfloat16):
                logits = model(inputs)

                #Reshaping logits and targets for loss function
                logits = logits.reshape([512,50257])
                targets = targets.flatten()
                loss = loss_fn(logits, targets)

                running_loss += loss.item()
        
    model.train()
    return running_loss / steps

def run_training(steps):
    running_loss = 0
    step = 0
    for i in range(steps):
        running_loss += train_step()
        step += 1
        if (step % 10 == 0):
            print(f"Step {step} average loss: {running_loss / 10}")
            running_loss = 0
        if (step % 20 == 0):
            print(f"Step {step} Eval Loss: {evaluate(5)}")

run_training(100)