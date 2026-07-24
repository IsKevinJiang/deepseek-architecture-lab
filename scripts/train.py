import torch
import torch.nn as nn 
from modelforge.model import Model
from modelforge.data import ShardDataLoader
from tqdm import trange
import math 
from pathlib import Path
from dataclasses import dataclass, asdict
import json

@dataclass
class TrainingConfig():
    #Model State
    hidden_dim: int = 128
    num_heads: int = 4
    sequence_length: int = 128
    batch_size: int = 4
    intermediate_dim: int = 512
    vocab_size : int =  50257
    num_layers : int = 4

    #Training parameters
    data_path : str = "data/fineweb_smoke"
    seed : int = 10
    min_lr : float = 3e-5
    max_lr: float = 3e-4
    weight_decay : float = 0.01
    warmup_steps : int = 100
    max_grad_norm : float = 1.0
    accumulation_steps: int = 4
    total_steps : int = 1120

    # Other (helps to log metrics)
    log_interval : int = 10
    eval_interval : int = 20
    eval_steps : int = 5
    checkpoint_interval : int = 200
    checkpoint_path : str = "checkpoints/latest.pt"
    run_dir: str = "runs/baseline"
    resume_training: bool = False

    #torch.compile (for improved speed)
    compile_model: bool = True

    def __post_init__(self):
        if self.num_heads <= 0:
            raise ValueError("num_heads is nonpositive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim is nonpositive")
        if self.batch_size <= 0:
            raise ValueError("batch_size is nonpositive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length is nonpositive")
        if self.max_lr <= 0:
            raise ValueError("max_lr is nonpositive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay is negative")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.min_lr > self.max_lr or self.min_lr <= 0:
            raise ValueError("min_lr not in range of [0, max_lr]")
        if (self.warmup_steps <= 0 or self.warmup_steps >= self.total_steps):
            raise ValueError("warmup_steps not in range of [0, total_steps]")
        if (self.max_grad_norm <= 0):
            raise ValueError("max_grad_norm is nonpositive")
        if (self.log_interval <= 0):
            raise ValueError("log_interval is nonpositive")
        if (self.eval_interval <= 0):
            raise ValueError("eval_interval is nonpositive")
        if (self.eval_steps <= 0):
            raise ValueError("eval_steps is nonpositive")
        if (self.accumulation_steps <= 0):
            raise ValueError("accumulation_steps is nonpositive")
        if (self.total_steps <= 0):
            raise ValueError("total_steps is nonpositive")
        if (self.checkpoint_interval <= 0):
            raise ValueError("checkpoint_interval is nonpositive")


config = TrainingConfig(
    hidden_dim=384,
    num_heads=6,
    intermediate_dim=1536,
    num_layers=6,
    sequence_length=128,
    batch_size=4,
    total_steps=29295,
    checkpoint_path="checkpoints/fineweb_60m_53m.pt",
    data_path="data/fineweb_60m",
    run_dir= "runs/fineweb_60m_53m",
    resume_training=False,
    eval_interval=500,
    eval_steps=50,
    warmup_steps=600,
    checkpoint_interval=1000,
    )

def initialize_run(config):
    run_path = Path(config.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    config_path = run_path / "config.json"
    config_data = asdict(config)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config_data, file, indent=2)

    metrics_path = run_path / "metrics.jsonl"
    return metrics_path

def append_metrics(metrics_path, record):
    with metrics_path.open("a", encoding="utf-8") as file:
        json.dump(record, file)
        file.write("\n")

def configure_optimizer(model, config):
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
            "weight_decay": config.weight_decay,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        }
    ]
    return torch.optim.AdamW(optimzer_groups, lr=config.max_lr)


def initialize_runtime(config):
    global device, model, train, val, val_state, optimizer, loss_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (not torch.cuda.is_bf16_supported() or not torch.cuda.is_available()):
        raise RuntimeError("No GPU or BF16 supported")

    torch.manual_seed(config.seed)
    model = Model(
        config.hidden_dim,
        config.num_heads,
        config.sequence_length,
        config.intermediate_dim,
        config.vocab_size,
        config.num_layers
    )
    model.to(device)

    train = ShardDataLoader(
        config.data_path,
        config.batch_size,
        config.sequence_length,
        "train",
        config.seed,
    )
    val = ShardDataLoader(
        config.data_path,
        config.batch_size,
        config.sequence_length,
        "val",
        config.seed,
    )
    val_state = val.state_dict()

    optimizer = configure_optimizer(model, config)
    loss_fn = nn.CrossEntropyLoss()


def train_step(config):
    if (config.accumulation_steps <= 0):
        raise ValueError("Accumulation steps should be positive")
    
    model.train()
    optimizer.zero_grad()
    running_loss = 0
    max_norm = config.max_grad_norm
    # Takes a batch of data and sends it to GPU
    for _ in range(config.accumulation_steps):
        inputs, targets = train.next_batch() #Both are[B, S]
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.autocast(device_type= "cuda", dtype=torch.bfloat16):
            logits = model(inputs) #[B, S, V]

            #Reshaping logits and targets for loss function
            logits = logits.reshape(-1, logits.size(-1)) # [B x S, V]
            targets = targets.flatten()
            batch_loss = loss_fn(logits, targets)
            running_loss += batch_loss.item()
            scaled_loss = batch_loss / config.accumulation_steps

        scaled_loss.backward()

    original_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm) #Gradient clipping 
    optimizer.step()
    average_loss = running_loss / config.accumulation_steps
    return average_loss, original_norm.item()

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
                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.flatten()
                loss = loss_fn(logits, targets)

                running_loss += loss.item()
        
    model.train()
    return running_loss / steps

def run_training(config, metrics_path, start_step=1):
    running_loss = 0.0
    latest_val_loss = None
    latest_avg_loss = None
    progress = trange(start_step, config.total_steps + 1, initial=start_step -1, total = config.total_steps, desc="Training", unit="step")

    for step in progress:
        #Updating the learning rate for each paramater group
        lr = scheduler(step, config)
        for param in optimizer.param_groups:
            param["lr"] = lr

        step_loss, original_norm = train_step(config)
        current_val = None

        #Calculating loss metrics
        running_loss += step_loss
        if step % config.log_interval == 0:
            latest_avg_loss = running_loss / config.log_interval
            running_loss = 0.0
        if step % config.eval_interval == 0:
            latest_val_loss = evaluate(config.eval_steps)
            current_val = latest_val_loss

        metrics = {
            "gradient_norm": f"{original_norm:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        }

        if latest_avg_loss is not None:
            metrics["average_loss"] = f"{latest_avg_loss:.4f}"
        if latest_val_loss is not None:
            metrics["val_loss"] = f"{latest_val_loss:.4f}"

        #Creating jsons for plotting
        progress.set_postfix(metrics)
        record = {
            "step" : step,
            "tokens": config.accumulation_steps * (config.batch_size * config.sequence_length) * step,
            "train_loss" : step_loss,
            "val_loss" : current_val,
            "learning_rate" : lr,
            "gradient_norm" : original_norm
        }
        append_metrics(metrics_path, record)

        #Saving the state in training for resuming training
        if(step % config.checkpoint_interval == 0 or step == config.total_steps):
            save_checkpoint(config.checkpoint_path, step)



def scheduler(current_step, config):
    if current_step <= config.warmup_steps:
        return current_step * (config.max_lr / config.warmup_steps)
    elif current_step >= config.total_steps:
        return config.min_lr
    else:
        #Cosine decay
        decay_progress = (current_step - config.warmup_steps) / (config.total_steps - config.warmup_steps)
        coefficient = 0.5 * (1 + math.cos(math.pi * decay_progress))
        return config.min_lr + coefficient * (config.max_lr - config.min_lr)

def save_checkpoint(path, step):
    checkpoint = {
        "model_state" : model.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
        "train_loader_state" : train.state_dict(),
        "global_step" : step,
        "CPU_RNG_state" : torch.get_rng_state(),
        "CUDA_RNG_state": torch.cuda.get_rng_state_all(),
    }
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(final_path)

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    train.load_state_dict(checkpoint["train_loader_state"])
    step = checkpoint["global_step"]
    torch.set_rng_state(checkpoint["CPU_RNG_state"])
    torch.cuda.set_rng_state_all(checkpoint["CUDA_RNG_state"])

    return step

def sync_metrics(metrics_path, saved_step):
    if not metrics_path.exists():
        return
    records = []
    with metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                metric = json.loads(line)
                if (metric["step"] <= saved_step):
                    records.append(metric)

    temp_path = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file)
            file.write("\n")
    temp_path.replace(metrics_path)

if __name__ == "__main__":
    initialize_runtime(config)

    checkpoint_path = Path(config.checkpoint_path)
    if (config.resume_training):
        if (checkpoint_path.exists()):
            saved_step = load_checkpoint(config.checkpoint_path)
            print(f"Checkpoint: {checkpoint_path} is resuming from {saved_step}")
        else:
            raise FileNotFoundError("Checkpoint path not found")
    else:
        if (checkpoint_path.exists()):
            raise FileExistsError("Checkpoint already exists")
        else:
            saved_step = 0
            print(f"Fresh training loop is starting at {checkpoint_path}")
    metrics_path = initialize_run(config)
    sync_metrics(metrics_path, saved_step)
    
    if config.compile_model:
        model.compile()
    run_training(config, metrics_path, start_step=saved_step + 1)
