import torch
import torch.nn as nn 
from modelforge.model import Model
from modelforge.data import ShardDataLoader
from tqdm import trange

def configure_optimizer(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if (param.requires_grad):
            if (param.dim() > 1):
                decay.append(param)
            else:
                no_decay.append(param)

    decay_ids = {id(param) for param in decay}
    no_decay_ids = {id(param) for param in no_decay}
    trainable_ids = {id(param) for name, param in model.named_parameters() if param.requires_grad}
    if (len(decay) != len(decay_ids) or len(no_decay) != len(no_decay_ids)):
        raise ValueError("Duplicate ids error")
    if (decay_ids & no_decay_ids):
        raise ValueError("Overlap Error")
    if (decay_ids | no_decay_ids != trainable_ids):
        raise ValueError("Trainable IDs not made up of all decay and non-decay")

    optimzer_groups = [
        {
            "params": decay,
            "weight_decay": 0.01,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,

        }
    ]
    return torch.optim.AdamW(optimzer_groups, lr=3e-4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if (not torch.cuda.is_bf16_supported() or not torch.cuda.is_available()):
    raise RuntimeError("No GPU or BF16 supported")

torch.manual_seed(10)
model = Model(128, 4, 128, 512, 50257, 4)
model.to(device)

train = ShardDataLoader("data/fineweb_smoke", 4, 128, "train", 10)
val = ShardDataLoader("data/fineweb_smoke", 4, 128, "val", 10)
val_state = val.state_dict()

optimizer = configure_optimizer(model)
loss_fn = nn.CrossEntropyLoss()

def train_step():
    model.train()
    # Takes a batch of data and sends it to GPU
    inputs, targets = train.next_batch()
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    with torch.autocast(device_type= "cuda", dtype=torch.bfloat16):
        logits = model(inputs)

        #Reshaping logits and targets for loss function
        logits = logits.reshape([512,50257])
        targets = targets.flatten()
        loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()

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
    running_loss = 0.0
    latest_val_loss = None

    progress = trange(1, steps + 1, desc="Training", unit="step")

    for step in progress:
        step_loss = train_step()
        running_loss += step_loss

        if step % 10 == 0:
            average_loss = running_loss / 10
            running_loss = 0.0

        if step % 20 == 0:
            latest_val_loss = evaluate(5)

        metrics = {
            "train_loss": f"{step_loss:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        }

        if latest_val_loss is not None:
            metrics["val_loss"] = f"{latest_val_loss:.4f}"

        progress.set_postfix(metrics)
    
run_training(10000)