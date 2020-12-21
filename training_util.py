from collections import deque
import torch


def average_reward(memory: deque, num_trials: int) -> float:
    memory_size = len(memory)

    total_reward = 0
    trials_seen = 0
    for i in range(min(num_trials, memory_size)):
        _, _, reward, _, terminal = memory[memory_size-1-i]
        total_reward += reward
        if terminal:
            trials_seen += 1

    return total_reward / max(1, trials_seen)


def checkpoint_model(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    memory: deque,
    path: str,
):
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'memory': memory,
    }, path)


def load_checkpoint(path: str) -> (dict, dict):
    checkpoint = torch.load(path)
    return checkpoint['model_state_dict'], checkpoint['optimizer_state_dict'], checkpoint['memory']
