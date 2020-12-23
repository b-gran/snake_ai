import pygame
from enum import Enum, auto

from typing import Optional, Deque, List, Any, Callable

from Grid import Grid

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque

from environment import (
    ActionType,
    Direction,
    Environment,
)
from rendering import draw_grid, draw_gradients
from test_net import TestNet
from training_util import load_checkpoint, average_reward, checkpoint_model

MemoryType = (List[int], ActionType, float, List[int], bool)

GRID_SIZE = (10, 10)
CELL_SIZE = 40
LOGGING = True

# training parameters
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 50000
BATCH_SIZE = 40

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01

# unused, doesn't apply to ε-greedy exploration
EXPLORATION_DECAY = 0.99995


def log(*args):
    if LOGGING:
        print(*args)


class Net(nn.Module):

    @staticmethod
    def conv2d_size_out(size, kernel_size, stride, padding=0):
        return (size - kernel_size + 2 * padding) // stride + 1

    def __init__(self, observation_space: (int, int)):
        super(Net, self).__init__()

        # conv_output_width =  Net.conv2d_size_out(observation_space[0], 1, 1)
        conv_output_width =  Net.conv2d_size_out(
            Net.conv2d_size_out(
                Net.conv2d_size_out(observation_space[0], 1, 1),
                3, 1, 2
            ),
            2, 1, 1
        )
        conv_output_height = conv_output_width

        linear_inputs = conv_output_width * conv_output_height * 64

        self.sequential = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, len(ActionType)),
        )

    def forward(self, inp: torch.FloatTensor):
        return self.sequential(inp)


class Solver:

    memory: Deque['MemoryType']
    device: Optional[torch.device]
    loss_buffer: Deque[float]

    def __init__(
        self,
        observation_space: (int, int),
        checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
        is_test: bool = False,
    ):
        self.test = is_test
        self.device = device
        self.exploration_rate = EXPLORATION_MAX

        self.observation_space = observation_space
        self.action_space = len(ActionType)

        # self.policy_net = TestNet()
        # self.target_net = TestNet()

        self.policy_net = Net(observation_space).to(self.device)
        self.target_net = Net(observation_space).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # params from deepmind
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=0.00025,
            momentum=0.95,
            eps=0.01,
        )

        # self.optimizer = optim.RMSprop(self.policy_net.parameters())
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)

        if checkpoint is not None:
            model_state, optimizer_state, memory_state = load_checkpoint(checkpoint)
            self.policy_net.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
            self.policy_net.train()

            self.memory: Deque['MemoryType'] = deque(memory_state, maxlen=MEMORY_SIZE)

            print('Loaded checkpoint from', checkpoint)
        else:
            self.memory: Deque['MemoryType'] = deque(maxlen=MEMORY_SIZE)

        self.batch_num = 0
        self.cumulative_loss = 0
        self.loss_buffer = deque(maxlen=100)

        self.frame = 0

    def memory_append(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def predict(self, state: List[int]) -> torch.Tensor:
        return self.policy_net(torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0))

    def epsilon(self, frame: int) -> float:
        # eps = lambda x: max(0.1, eps_max - x*(eps_max-eps_min)/eps_frames)
        exploration_frames = 150000
        return max(
            EXPLORATION_MIN,
            EXPLORATION_MAX - frame * (EXPLORATION_MAX - EXPLORATION_MIN) / exploration_frames
        )

    # if not in test mode, returns (predicted action, None)
    # if in test mode, returns (predicted action, gradient of prediction with respect to input)
    def act(self, state: [int]) -> (ActionType, Optional[torch.tensor]):
        current_frame = self.frame
        self.frame += 1

        if not self.test:
            if random.random() < self.epsilon(current_frame):
                return ActionType(random.randrange(self.action_space)), None

            predictions = self.predict(state)
            return ActionType(int(predictions.argmax())), None

        input_leaf = torch.tensor(state, dtype=torch.float, device=self.device, requires_grad=True)
        input_tensor = input_leaf.unsqueeze(0)
        result = self.policy_net(input_tensor)

        # compute the gradient of the output with respect to the input
        result[0][result.argmax()].backward()
        return ActionType(int(result.argmax())), input_leaf.grad.squeeze()

    def print_state(self, state: np.ndarray):
        grid = np.reshape(state[0][:self.observation_space-1], GRID_SIZE)
        direction = state[0][self.observation_space-1]
        print(grid)
        print(Direction(direction))

    def do_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        if self.batch_num % 100 == 0:
            log('batch', self.batch_num)
            log('average loss', sum(self.loss_buffer) / max(len(self.loss_buffer), 1))
            log('exploration rate', self.exploration_rate)
            self.cumulative_loss = 0

        batch = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, state_nexts, terminals = zip(*batch)
        non_terminal_mask = torch.tensor(list(map(
            lambda is_terminal: not is_terminal,
            terminals
        )), device=self.device)
        non_terminal_next_states = torch.tensor([
            s for s, t in zip(state_nexts, terminals) if not t
        ], dtype=torch.float, device=self.device)

        action_tensor = torch.tensor(list(map(lambda a: a.value, actions)), device=self.device).reshape((BATCH_SIZE, 1))
        state_tensor = torch.tensor(states, dtype=torch.float, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)

        current_predictions = self.policy_net(state_tensor).gather(1, action_tensor)
        next_predictions = torch.zeros(BATCH_SIZE, device=self.device)
        next_predictions[non_terminal_mask] = self.target_net(non_terminal_next_states).max(1)[0].detach()
        expected_q_values = (next_predictions * GAMMA) + reward_tensor

        loss = F.smooth_l1_loss(current_predictions, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-5, 5)

        self.optimizer.step()

        self.loss_buffer.append(float(loss))

        self.exploration_rate = max(self.exploration_rate * EXPLORATION_DECAY, EXPLORATION_MIN)

        self.batch_num += 1


def train_loop(
    maybe_device: Optional[torch.device] = None,
    gui: bool = True,
    plot_stats: Callable[[int, Solver], None] = lambda _, __: None,
    parameter_update_count: int = 1000,
    checkpoint_update_count: int = 10000,
):
    render_visualization = True
    screen = None
    if gui:
        screen = pygame.display.set_mode((GRID_SIZE[0] * CELL_SIZE, GRID_SIZE[1] * CELL_SIZE))

    grid = Grid(GRID_SIZE[0], GRID_SIZE[1], CELL_SIZE)

    env = Environment(GRID_SIZE)
    solver = Solver(
        env.get_state().shape[1:],
        # checkpoint='model_10x10_400kbatch.pt',
        device=maybe_device,
    )

    action_count = 0

    while True:
        env.reset()
        state = env.get_state()

        while True:
            if action_count % checkpoint_update_count == 0:
                checkpoint_model(solver.policy_net, solver.optimizer, solver.memory, 'model.pt')

            if gui:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                            render_visualization = not render_visualization
                            continue

            action_count += 1

            # every COUNT actions, update the parameters in the target net
            # based on the policy net.
            if action_count % parameter_update_count == 0:
                solver.target_net.load_state_dict(solver.policy_net.state_dict())

            if action_count % 1000 == 0:
                log('average reward', average_reward(solver.memory, 1000))

            action, _ = solver.act(state)
            state_next, reward, terminal = env.step(action)

            solver.memory_append(state, action, reward, state_next, terminal)

            state = state_next

            if terminal:
                break

            solver.do_replay()

            if render_visualization:
                grid.draw_background(screen)
                draw_grid(env.grid, screen, CELL_SIZE)
                pygame.display.flip()

            if action_count % 5000 == 0:
                plot_stats(action, solver)


def human_test_loop():
    screen = pygame.display.set_mode((GRID_SIZE[0] * CELL_SIZE, GRID_SIZE[1] * CELL_SIZE))
    grid = Grid(GRID_SIZE[0], GRID_SIZE[1], CELL_SIZE)
    clock = pygame.time.Clock()

    env = Environment(GRID_SIZE)

    while True:
        clock.tick(60)

        terminal = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    _, reward, terminal = env.step(ActionType.ACTION_TYPE_UP)
                    print('reward', reward)
                elif event.key == pygame.K_RIGHT:
                    _, reward, terminal = env.step(ActionType.ACTION_TYPE_RIGHT)
                    print('reward', reward)
                elif event.key == pygame.K_DOWN:
                    _, reward, terminal = env.step(ActionType.ACTION_TYPE_DOWN)
                    print('reward', reward)
                elif event.key == pygame.K_LEFT:
                    _, reward, terminal = env.step(ActionType.ACTION_TYPE_LEFT)
                    print('reward', reward)

        if terminal:
            env.reset()

        grid.draw_background(screen)
        draw_grid(env.grid, screen, CELL_SIZE)
        pygame.display.flip()


def test(snapshot: str):
    assert len(snapshot) > 0, 'must provide a snapshot'
    screen = pygame.display.set_mode((
        GRID_SIZE[0] * CELL_SIZE * 2,
        GRID_SIZE[1] * CELL_SIZE
    ))
    clock = pygame.time.Clock()

    grid = Grid(GRID_SIZE[0], GRID_SIZE[1], CELL_SIZE)

    env = Environment(GRID_SIZE)
    solver = Solver(
        env.get_state().shape[1:],
        checkpoint=snapshot,
        is_test=True,
    )
    solver.exploration_rate = 0.01
    state = env.get_state()

    autoplay = True
    gradient = None
    is_terminated = False

    while True:
        clock.tick(20)

        run_simulation = autoplay

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    env.reset()
                    continue

                if event.key == pygame.K_UP:
                    autoplay = not autoplay

                if event.key == pygame.K_RIGHT and not autoplay:
                    run_simulation = True

        if run_simulation:
            if is_terminated:
                is_terminated = False
                gradient = None
                env.reset()
            else:
                action, gradient = solver.act(state)
                state_next, _, terminal = env.step(action)
                state = state_next
                if terminal:
                    is_terminated = True

        grid.draw_background(screen)
        draw_grid(env.grid, screen, CELL_SIZE)
        draw_gradients(gradient, screen, CELL_SIZE, (GRID_SIZE[0] * CELL_SIZE, 0))
        pygame.display.flip()


if __name__ == '__main__':
    train_loop()
    # test('gpu_10x10_deepmind_rms.pt')
    # human_test_loop()
