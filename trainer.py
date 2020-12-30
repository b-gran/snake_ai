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
from rendering import draw_grid, draw_gradients, draw_stats
from test_net import TestNet
from training_util import load_checkpoint, average_reward, checkpoint_model
from plotting_util import get_pyplot_plotter


torch.autograd.set_detect_anomaly(True)

MemoryType = (List[int], torch.tensor, ActionType, float, List[int], torch.tensor, bool)

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


class AttentionBlock(nn.Module):

    def __init__(self, input_features: int):
        super(AttentionBlock, self).__init__()

        self.operator = nn.Conv2d(
            in_channels=input_features,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=False,
        )

    def forward(self, l: torch.Tensor, g: torch.Tensor):
        batch_size, channels, width, height = l.size()
        g = g.reshape([batch_size, channels, 1, 1])
        c = self.operator(l + g)
        a = F.softmax(
            c.view(batch_size, 1, -1),
            dim=2
        ).view(batch_size, 1, width, height)  # reshape to square

        # weighted sum attention
        g = torch.mul(a.expand_as(l), l)

        # sum over grid to produce [batch_size, channels]
        g = g.view(batch_size, channels, -1).sum(dim=2)

        return c.view(batch_size, 1, width, height), g


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

        # self.sequential = nn.Sequential(
        #     nn.Conv2d(1, 8, kernel_size=1, stride=1),
        #     nn.ReLU(),
        #     # attention here
        #     nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(linear_inputs, 512),
        #     nn.ReLU(),
        #     # attention here
        #     nn.Linear(512, len(ActionType)),
        # )

        self.c1 = nn.Conv2d(1, 8, kernel_size=1, stride=1)
        # self.a1 = AttentionBlock(8)
        # self.a1 = AttentionBlock(512)

        self.c2 = nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1)
        # self.a2 = AttentionBlock(64)
        self.a2 = AttentionBlock(256)

        self.dense = nn.Linear(linear_inputs, 256)

        self.out = nn.Linear(512, len(ActionType))
        # self.out = nn.Linear(72, len(ActionType))

        # self.proj1 = nn.Linear(512, 8)
        # self.conv_proj1 = nn.Conv2d(8, 512, kernel_size=1, stride=1, padding=0)

        # self.proj2 = nn.Linear(256, 64)
        self.conv_proj2 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

        hidden_size = 512
        self.lstm = nn.LSTM(512, hidden_size)

        initial_hidden_tensor = nn.Parameter(torch.randn(2, hidden_size), requires_grad=True)
        # initial_hidden_context = nn.Parameter(torch.randn(hidden_size), requires_grad=True)

        # self.initial_hidden = nn.Parameter(initial_hidden_tensor, requires_grad=True)

        # self.initial_hidden = nn.Parameter(initial_hidden_tensor, requires_grad=True)
        # self.initial_hidden = torch.randn(2, hidden_size)
        self.initial_hidden = initial_hidden_tensor

    def forward(self, game_input: torch.FloatTensor, prev_state: torch.FloatTensor):
        l1 = F.relu(self.c1(game_input))
        x = F.relu(self.c2(l1))
        l2 = F.relu(self.c3(x))
        x = l2.flatten(1)
        g = F.relu(self.dense(x))

        # c1, g1 = self.a1(l1, self.proj1(g))
        # c1, g1 = self.a1(self.conv_proj1(l1), g)

        # c2, g2 = self.a2(l2, self.proj2(g))
        c2, g2 = self.a2(self.conv_proj2(l2), g)

        all_attention = torch.cat((g, g2), dim=1)

        attn_seq = all_attention.unsqueeze(0)

        # dim 0 is (hidden_state, ctx) so add sequence at dimension 1
        prev_state_seq = prev_state.unsqueeze(1)
        prev_state = (prev_state_seq[0].detach(), prev_state_seq[1].detach())

        lstm_out, hidden = self.lstm(
            attn_seq,
            prev_state,
        )

        # x = self.out(all_attention)
        x = self.out(lstm_out)
        return x, hidden


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
        exploration_frames: int = 50000,
        exploration_max: float = EXPLORATION_MAX,
        exploration_min: float = EXPLORATION_MIN,
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
        self.loss_buffer = deque(maxlen=5000)
        self.prediction_buffer = deque(maxlen=5000)

        self.frame = 0

        self.exploration_frames = exploration_frames
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min

        self.last_hidden_state = self.policy_net.initial_hidden

    def memory_append(self, state, hidden, action, reward, next_state, next_hidden, done):
        self.memory.append((state, hidden, action, reward, next_state, next_hidden, done))

    def predict(self, state: List[int]) -> torch.Tensor:
        return self.policy_net(torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0))

    def epsilon(self, frame: int) -> float:
        # eps = lambda x: max(0.1, eps_max - x*(eps_max-eps_min)/eps_frames)
        return max(
            self.exploration_min,
            self.exploration_max - frame * (self.exploration_max - self.exploration_min) / self.exploration_frames
        )

    def reset_hidden_state(self):
        self.last_hidden_state = self.policy_net.initial_hidden

    # if not in test mode, returns (predicted action, None)
    # if in test mode, returns (predicted action, hidden state, gradient of prediction with respect to input)
    def act(self, state: [int]) -> (ActionType, Optional[torch.tensor]):
        current_frame = self.frame
        self.frame += 1

        if not self.test:
            if random.random() < self.epsilon(current_frame):
                return ActionType(random.randrange(self.action_space)), None

            # add batch dimension
            input_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)

            # first dimension is (state, ctx), so add batch at dim 1
            hidden_state_batch = self.last_hidden_state.unsqueeze(1)

            predictions, hidden = self.policy_net(input_tensor, hidden_state_batch)

            # predictions = self.predict(state)

            # first dimension should be (state, ctx)
            # squeeze out batch and sequence dimensions (sequence is always 1)
            self.last_hidden_state = torch.stack((
                hidden[0].squeeze(),
                hidden[1].squeeze(),
            ))
            self.prediction_buffer.append(float(predictions.max()))

            return ActionType(int(predictions.argmax())), None

        input_leaf = torch.tensor(state, dtype=torch.float, device=self.device, requires_grad=True)
        input_tensor = input_leaf.unsqueeze(0)
        result, hidden = self.policy_net(input_tensor, self.last_hidden_state.unsqueeze(0))

        self.last_hidden_state = hidden

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

        states, hiddens, actions, rewards, state_nexts, next_hiddens, terminals = zip(*batch)
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

        # stack on dim=1 so that dim 0 is (state, ctx)
        hidden_tensor = torch.stack(hiddens, 1).to(self.device)

        non_terminal_next_hiddens = torch.stack([
            s for s, t in zip(next_hiddens, terminals) if not t
        ], 1).to(self.device)

        # original
        # current_predictions = self.policy_net(state_tensor, hidden_tensor).gather(1, action_tensor)

        out, _ = self.policy_net(state_tensor, hidden_tensor)
        current_predictions = out.squeeze().gather(1, action_tensor)  # squeeze to remove sequence dimension

        next_predictions = torch.zeros(BATCH_SIZE, device=self.device)

        # original
        # next_predictions[non_terminal_mask] = self.target_net(non_terminal_next_states, non_terminal_next_hiddens).max(1)[0].detach()

        next_out, _ = self.target_net(non_terminal_next_states, non_terminal_next_hiddens)
        next_predictions[non_terminal_mask] = next_out.detach().squeeze().max(1)[0]

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
    plot_stats: Callable[[int, Solver, Deque], None] = lambda _, __: None,
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
        exploration_frames=1,
    )

    action_count = 0
    scores = deque()
    current_score = 0

    while True:
        env.reset()
        state = env.get_state()
        solver.reset_hidden_state()

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

            hidden = solver.last_hidden_state
            action, _ = solver.act(state)
            next_hidden = solver.last_hidden_state
            state_next, reward, terminal = env.step(action)

            solver.memory_append(state, hidden.clone(), action, reward, state_next, next_hidden.clone(), terminal)

            state = state_next

            if reward > 0:
                current_score += 1

            if terminal:
                scores.append(current_score)
                current_score = 0
                break

            solver.do_replay()

            if gui and render_visualization:
                grid.draw_background(screen)
                draw_grid(env.grid, screen, CELL_SIZE)
                pygame.display.flip()

            if action_count % 5000 == 0:
                plot_stats(action_count, solver, scores)
                scores.clear()


def human_test_loop():
    screen = pygame.display.set_mode((GRID_SIZE[0] * CELL_SIZE, GRID_SIZE[1] * CELL_SIZE))
    grid = Grid(GRID_SIZE[0], GRID_SIZE[1], CELL_SIZE)
    clock = pygame.time.Clock()

    env = Environment(GRID_SIZE)
    finished_episode = False

    while True:
        clock.tick(60)

        terminal = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if finished_episode:
                    finished_episode = False
                    env.reset()
                elif event.key == pygame.K_UP:
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

        grid.draw_background(screen)
        draw_grid(env.grid, screen, CELL_SIZE)
        pygame.display.flip()

        if terminal:
            finished_episode = True

def test(snapshot: str):
    assert len(snapshot) > 0, 'must provide a snapshot'
    pygame.init()
    screen = pygame.display.set_mode((
        GRID_SIZE[0] * CELL_SIZE * 2,
        GRID_SIZE[1] * CELL_SIZE + 25
    ))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('americantypewriter', 14)

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
    current_score = 0
    cumulative_score = 0
    n_episodes = 1

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
                n_episodes += 1
                current_score = 0
                env.reset()
            else:
                action, gradient = solver.act(state)
                state_next, reward, terminal = env.step(action)
                state = state_next
                if reward > 0:
                    current_score += 1
                    cumulative_score += 1
                if terminal:
                    is_terminated = True

        screen.fill((0, 0, 0))
        grid.draw_background(screen)
        draw_grid(env.grid, screen, CELL_SIZE)
        draw_gradients(gradient, screen, CELL_SIZE, (GRID_SIZE[0] * CELL_SIZE, 0))
        draw_stats(
            screen,
            (0, GRID_SIZE[1] * CELL_SIZE + 5),
            int(clock.get_fps()),
            current_score,
            cumulative_score / n_episodes,
            font,
        )
        pygame.display.flip()


if __name__ == '__main__':
    # train_loop(plot_stats=get_pyplot_plotter())
    # test('10x10_1M_512_smooth.pt')
    human_test_loop()
