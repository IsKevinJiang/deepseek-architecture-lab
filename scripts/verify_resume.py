import copy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts import train as training


def clone_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def assert_state_equal(expected, actual, path="state"):
    if isinstance(expected, torch.Tensor):
        actual_tensor = actual.detach().cpu()
        if not torch.equal(expected, actual_tensor):
            raise AssertionError(f"Tensor mismatch at {path}")
        return

    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"Dictionary keys differ at {path}")
        for key in expected:
            assert_state_equal(expected[key], actual[key], f"{path}.{key}")
        return

    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise AssertionError(f"Sequence length differs at {path}")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            assert_state_equal(expected_item, actual_item, f"{path}[{index}]")
        return

    if expected != actual:
        raise AssertionError(f"Value mismatch at {path}: {expected!r} != {actual!r}")


def run_steps(start_step, end_step, config):
    metrics = []
    for step in range(start_step, end_step + 1):
        learning_rate = training.scheduler(step, config)
        for group in training.optimizer.param_groups:
            group["lr"] = learning_rate

        loss, gradient_norm = training.train_step(config)
        metrics.append((loss, gradient_norm, learning_rate))
    return metrics


def verify_resume(
    total_steps=12,
    split_step=6,
    warmup_steps=4,
    accumulation_steps=4,
):
    if not 0 < split_step < total_steps:
        raise ValueError("split_step must be between zero and total_steps")
    if not 0 < warmup_steps < total_steps:
        raise ValueError("warmup_steps must be between zero and total_steps")

    test_config = replace(
        training.config,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        accumulation_steps=accumulation_steps,
    )

    checkpoint_root = Path("checkpoints")
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".resume-check-", dir=checkpoint_root) as temp_dir:
        checkpoint_path = Path(temp_dir) / "state.pt"

        # Both experiments begin from this exact model, optimizer, loader, and RNG state.
        training.save_checkpoint(checkpoint_path, step=0)

        uninterrupted_metrics = run_steps(1, total_steps, test_config)
        uninterrupted_model = clone_to_cpu(training.model.state_dict())
        uninterrupted_optimizer = clone_to_cpu(training.optimizer.state_dict())
        uninterrupted_loader = clone_to_cpu(training.train.state_dict())
        uninterrupted_next_batch = tuple(tensor.clone() for tensor in training.train.next_batch())
        uninterrupted_cpu_random = torch.rand(8)
        uninterrupted_cuda_random = torch.rand(8, device=training.device).cpu()

        initial_step = training.load_checkpoint(checkpoint_path)
        if initial_step != 0:
            raise AssertionError(f"Expected initial step 0, received {initial_step}")

        resumed_prefix = run_steps(1, split_step, test_config)
        training.save_checkpoint(checkpoint_path, step=split_step)

        # Mutate every live state before loading to prove restoration is doing real work.
        run_steps(split_step + 1, split_step + 1, test_config)
        torch.rand(8)
        torch.rand(8, device=training.device)

        restored_step = training.load_checkpoint(checkpoint_path)
        if restored_step != split_step:
            raise AssertionError(
                f"Expected restored step {split_step}, received {restored_step}"
            )

        resumed_suffix = run_steps(
            restored_step + 1,
            total_steps,
            test_config,
        )
        resumed_metrics = resumed_prefix + resumed_suffix

        assert_state_equal(uninterrupted_metrics, resumed_metrics, "metrics")
        assert_state_equal(
            uninterrupted_model,
            training.model.state_dict(),
            "model_state",
        )
        assert_state_equal(
            uninterrupted_optimizer,
            training.optimizer.state_dict(),
            "optimizer_state",
        )
        assert_state_equal(
            uninterrupted_loader,
            training.train.state_dict(),
            "train_loader_state",
        )

        resumed_next_batch = training.train.next_batch()
        assert_state_equal(
            uninterrupted_next_batch,
            resumed_next_batch,
            "next_batch",
        )

        resumed_cpu_random = torch.rand(8)
        resumed_cuda_random = torch.rand(8, device=training.device).cpu()
        assert_state_equal(
            uninterrupted_cpu_random,
            resumed_cpu_random,
            "CPU_RNG_state",
        )
        assert_state_equal(
            uninterrupted_cuda_random,
            resumed_cuda_random,
            "CUDA_RNG_state",
        )

    print(
        "Resume verification passed: "
        f"{total_steps} uninterrupted updates exactly matched "
        f"{split_step} + {total_steps - split_step} resumed updates "
        f"with {accumulation_steps} accumulation steps."
    )


if __name__ == "__main__":
    verify_resume()
