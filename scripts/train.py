import torch
from modelforge.model import Model
from modelforge.data import ShardDataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Model(128, 4, 128, 512, 50257, 4)
train = ShardDataLoader("data/fineweb_smoke", 4, 128, "train", 10)
val = ShardDataLoader("data/fineweb_smoke", 4, 128, "val", 10)

#Data is going from uint16 to int64
train_input, train_output = train.next_batch()
val_input, val_output = val.next_batch()

#Checking dimensions, tensor on cpu, and datatype is int64 (size of torch.long)
if (train_input.shape != torch.Size([4,128]) or train_input.get_device() != -1  or train_input.dtype != torch.int64):
    raise ValueError("Training input batch is invalid")
if (train_output.shape != torch.Size([4,128]) or train_output.get_device() != -1 or train_output.dtype != torch.int64):
    raise ValueError("Training input batch is invalid")
if (val_input.shape != torch.Size([4,128]) or val_input.get_device() != -1 or val_input.dtype != torch.int64):
    raise ValueError("Validation input batch is invalid")
if (val_output.shape != torch.Size([4,128]) or val_output.get_device() != -1 or val_output.dtype != torch.int64):
    raise ValueError("Validation output batch is invalid")

