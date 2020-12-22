import pygame
import random
from enum import Enum, auto

from typing import Optional, Deque, List, Any
from nptyping import NDArray

from Grid import Grid

import numpy as np

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque

from training_util import load_checkpoint, average_reward, checkpoint_model

random.seed(1011970)

GRID_SIZE = (3, 3)
CELL_SIZE = 40

CELL_TYPE_EMPTY = 0
CELL_TYPE_HEAD = 1
CELL_TYPE_BODY = 2
CELL_TYPE_FOOD = 3

# FFA1C7
COLOR_FOOD = (255, 161, 199)


class ActionType(Enum):
    ACTION_TYPE_NOTHING = 0
    ACTION_TYPE_LEFT = 1
    ACTION_TYPE_UP = 2
    ACTION_TYPE_RIGHT = 3
    ACTION_TYPE_DOWN = 4


Position = (int, int)


class BodyNode:
    position: Position
    next: Optional['BodyNode']
    prev: Optional['BodyNode']

    def __init__(self, position: Position):
        self.position = position
        self.next = None
        self.prev = None


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


DIRECTION_BY_ACTION = {
    ActionType.ACTION_TYPE_LEFT: Direction.LEFT,
    ActionType.ACTION_TYPE_RIGHT: Direction.RIGHT,
    ActionType.ACTION_TYPE_UP: Direction.UP,
    ActionType.ACTION_TYPE_DOWN: Direction.DOWN,
}

LOGGING = False


def log(*args):
    if LOGGING:
        print(*args)


def move_direction(position: Position, direction: Direction) -> Position:
    r, c = position
    if direction == Direction.UP:
        return r - 1, c

    if direction == Direction.RIGHT:
        return r, c + 1

    if direction == Direction.DOWN:
        return r + 1, c

    if direction == Direction.LEFT:
        return r, c - 1

    raise Exception('Invalid direction')


EnvironmentGrid = List[List[int]]
TState = NDArray[(Any, Any), np.float]


def add_position(a: Position, b: Position) -> Position:
    return a[0] + b[0], a[1] + b[1]


class Environment:
    body_tail: BodyNode
    body_head: BodyNode
    grid: EnvironmentGrid
    head_position: Position
    food_position: Position
    direction: Direction
    body_length: int
    state_space: int

    @staticmethod
    def blank_state(size: (int, int)):
        return [
            [CELL_TYPE_EMPTY] * size[1] for _ in range(size[0])
        ]

    def __init__(self, size: (int, int)):
        self.size = size
        self.num_cells = size[0] * size[1]
        self.state_space = self.num_cells + 1
        self.reset()

    def copy_grid(self):
        copy = Environment.blank_state(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                copy[i][j] = self.grid[i][j]
        return copy

    def get_state(self) -> TState:
        np_float_grid = np.reshape(self.grid, self.size) * 1.0
        return np.expand_dims(np_float_grid, 0)

    def get_new_food_position(self):
        valid_positions = []
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                if self.grid[r][c] == CELL_TYPE_EMPTY:
                    valid_positions.append((r, c))

        if len(valid_positions) == 0:
            raise Exception('No valid positions on grid')

        return random.choice(valid_positions)

    def reset(self):
        # Init empty state
        rows = self.size[0]
        cols = self.size[1]
        self.grid = Environment.blank_state(self.size)
        for i in range(rows):
            for j in range(cols):
                self.grid[i][j] = 0

        # Init body
        self.head_position = (int(rows / 2), int(cols / 2))
        self.grid[self.head_position[0]][self.head_position[1]] = CELL_TYPE_HEAD
        self.body_head = BodyNode(self.head_position)
        self.body_tail = self.body_head
        self.body_length = 1

        # Init food
        self.food_position = self.get_new_food_position()
        self.grid[self.food_position[0]][self.food_position[1]] = CELL_TYPE_FOOD

        # Init direction
        self.direction = Direction.RIGHT

    # Returns (reward, is_terminal)
    def get_reward(self, new_position: Position) -> (float, bool):
        r, c = new_position

        if (
            (
                r >= self.size[0] or
                r < 0 or
                c >= self.size[1] or
                c < 0 or
                self.grid[r][c] == CELL_TYPE_BODY or
                self.grid[r][c] == CELL_TYPE_HEAD
            ) and not (
                self.body_tail.position[0] == r and
                self.body_tail.position[1] == c
            )
        ):
            return -10.0, True

        if (
            self.body_length == (self.size[0] * self.size[1]) - 1 and
            self.grid[r][c] == CELL_TYPE_FOOD
        ):
            return 1000.0, True

        return (
            100.0 if self.grid[r][c] == CELL_TYPE_FOOD else 0.0,
            False
        )

    def is_legal_position(self, position: Position) -> bool:
        r, c = position
        return not (r >= self.size[0] or r < 0 or c >= self.size[1] or c < 0 or self.grid[r][c] == CELL_TYPE_BODY)

    def can_move(self, action: ActionType) -> bool:
        if action not in DIRECTION_BY_ACTION:
            return False

        r, c = move_direction(self.head_position, DIRECTION_BY_ACTION[action])

        if (
            (
                r >= self.size[0] or
                r < 0 or
                c >= self.size[1] or
                c < 0 or
                self.grid[r][c] == CELL_TYPE_BODY
            ) and not
            (
                self.body_tail.position[0] == r and self.body_tail.position[1] == c
            )
        ):
            return False

        return True

    def step(self, action: ActionType) -> (TState, float, bool):
        if self.can_move(action):
            self.direction = DIRECTION_BY_ACTION[action]

        new_position = move_direction(self.head_position, self.direction)

        reward, is_terminal = self.get_reward(new_position)
        if is_terminal:
            return self.get_state(), reward, is_terminal

        self.grid = self.copy_grid()

        # Figure out if we ate food and need to grow
        ate_food = self.grid[new_position[0]][new_position[1]] == CELL_TYPE_FOOD

        # Move the head in the direction
        self.grid[self.head_position[0]][self.head_position[1]] = CELL_TYPE_BODY
        self.grid[new_position[0]][new_position[1]] = CELL_TYPE_HEAD
        self.head_position = new_position

        # Need to grow the snake
        if ate_food:
            self.body_length += 1

            # Create new head
            new_head = BodyNode(new_position)

            # Add in front of current head
            self.body_head.next = new_head
            new_head.prev = self.body_head

            # Update head
            self.body_head = new_head

            # Place new food
            new_food_position = self.get_new_food_position()
            self.grid[new_food_position[0]][new_food_position[1]] = CELL_TYPE_FOOD
        else:
            # Not growing snake
            prev_tail = self.body_tail

            # Clear cell at previous tail if we're not moving onto the tail
            r, c = prev_tail.position
            if self.grid[r][c] != CELL_TYPE_HEAD:
                self.grid[r][c] = CELL_TYPE_EMPTY

            if prev_tail.next:
                # More than one node in the body, need to remove the tail.

                # Clean up tail
                prev_tail.next.prev = None
                self.body_tail = prev_tail.next

                # Add prev tail at head
                self.body_head.next = prev_tail
                prev_tail.prev = self.body_head
                self.body_head = prev_tail
                self.body_head.next = None

            # Update head
            self.body_head.position = new_position

        return self.get_state(), reward, is_terminal

    def has_valid_moves(self) -> bool:
        potential_moves = [
            add_position(self.head_position, (-1, 0)),  # up
            add_position(self.head_position, (0, 1)),  # right
            add_position(self.head_position, (1, 0)),  # down
            add_position(self.head_position, (0, -1)),  # left
        ]

        for m in potential_moves:
            if self.is_legal_position(m):
                return True

        return False


def draw_grid(state: EnvironmentGrid, surface: pygame.Surface, cell_size: int):
    rows = len(state)
    cols = len(state[0])
    for i in range(rows):
        for j in range(cols):
            if state[i][j] == CELL_TYPE_BODY or state[i][j] == CELL_TYPE_HEAD:
                color = (255, 100, 100) if state[i][j] == CELL_TYPE_BODY else (200, 80, 80)
                pygame.draw.rect(
                    surface,
                    color,
                    [
                        j * cell_size,
                        i * cell_size,
                        cell_size,
                        cell_size,
                    ]
                )

            if state[i][j] == CELL_TYPE_FOOD:
                pygame.draw.circle(
                    surface, COLOR_FOOD, [j * cell_size + cell_size / 2, i * cell_size + cell_size / 2],
                    int(cell_size / 3)
                )


GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 50000
BATCH_SIZE = 40

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.9995


MemoryType = (List[int], ActionType, float, List[int], bool)


class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            # nn.Linear(64, len(ActionType)),
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(ActionType)),
        )

    def forward(self, x):
        # result = self.conv(x)
        y1 = self.c1(x)
        y2 = self.c2(y1)
        y3 = self.flatten(y2)
        return y3


class Solver:

    memory: Deque['MemoryType']

    def __init__(
        self,
        observation_space: (int, int),
        checkpoint: Optional[str] = None,
    ):
        self.exploration_rate = EXPLORATION_MAX

        self.observation_space = observation_space
        self.action_space = len(ActionType)

        linear_inputs = (observation_space[0]+1) * (observation_space[1]+1) * 4

        # self.model = TestNet()

        # working conv
        self.policy_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(ActionType)),
        )

        # "old" network used for evaluating next states
        self.target_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(ActionType)),
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
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

    def memory_append(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def predict(self, state: List[int]) -> torch.Tensor:
        return self.policy_net(torch.tensor(state, dtype=torch.float).unsqueeze(0))

    def act(self, state: [int]) -> ActionType:
        if random.random() < self.exploration_rate:
            return ActionType(random.randrange(self.action_space))

        predictions = self.predict(state)

        return ActionType(int(predictions.argmax()))

    def print_state(self, state: NDArray[(Any, Any), np.int]):
        grid = np.reshape(state[0][:self.observation_space-1], GRID_SIZE)
        direction = state[0][self.observation_space-1]
        print(grid)
        print(Direction(direction))

    def do_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        if self.batch_num % 100 == 0:
            print('batch', self.batch_num)
            print('average loss', self.cumulative_loss / 100)
            print('exploration rate', self.exploration_rate)
            self.cumulative_loss = 0

        batch = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, state_nexts, terminals = zip(*batch)
        non_terminal_mask = torch.tensor(list(map(
            lambda is_terminal: not is_terminal,
            terminals
        )))
        non_terminal_next_states = torch.tensor([
            s for s, t in zip(state_nexts, terminals) if not t
        ], dtype=torch.float)

        action_tensor = torch.tensor(list(map(lambda a: a.value, actions))).reshape((BATCH_SIZE, 1))
        state_tensor = torch.tensor(states, dtype=torch.float)
        reward_tensor = torch.tensor(rewards, dtype=torch.float)

        current_predictions = self.policy_net(state_tensor).gather(1, action_tensor)
        next_predictions = torch.zeros(BATCH_SIZE)
        next_predictions[non_terminal_mask] = self.target_net(non_terminal_next_states).max(1)[0].detach()
        expected_q_values = (next_predictions * GAMMA) + reward_tensor

        loss = F.smooth_l1_loss(current_predictions, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-5, 5)

        self.optimizer.step()

        self.cumulative_loss += float(loss)

        self.exploration_rate = max(self.exploration_rate * EXPLORATION_DECAY, EXPLORATION_MIN)

        self.batch_num += 1


def train_loop():
    screen = pygame.display.set_mode((GRID_SIZE[0] * CELL_SIZE, GRID_SIZE[1] * CELL_SIZE))
    grid = Grid(GRID_SIZE[0], GRID_SIZE[1], CELL_SIZE)
    clock = pygame.time.Clock()

    env = Environment(GRID_SIZE)
    solver = Solver(
        env.get_state().shape[1:],
        # checkpoint='model_bad_2.pt',
    )
    state = env.get_state()

    do_visualize = True
    action_count = 0
    parameter_update_count = 500

    while True:
        env.reset()
        state = env.get_state()

        while True:
            if do_visualize:
                clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            action_count += 1

            # every COUNT actions, update the parameters in the target net
            # based on the policy net.
            if action_count % parameter_update_count == 0:
                solver.target_net.load_state_dict(solver.policy_net.state_dict())

            if action_count % 1000 == 0:
                print('average reward', average_reward(solver.memory, 1000))

            action = solver.act(state)
            state_next, reward, terminal = env.step(action)

            solver.memory_append(state, action, reward, state_next, terminal)

            state = state_next

            if terminal:
                log('terminal', reward)
                break

            solver.do_replay()

            if do_visualize:
                grid.draw_background(screen)
                draw_grid(env.grid, screen, CELL_SIZE)
                pygame.display.flip()


def human_test_loop():
    screen = pygame.display.set_mode((GRID_SIZE[0] * CELL_SIZE, GRID_SIZE[1] * CELL_SIZE))
    grid = Grid(GRID_SIZE[0], GRID_SIZE[1], CELL_SIZE)
    clock = pygame.time.Clock()

    env = Environment(GRID_SIZE)

    while True:
        clock.tick(60)

        terminal = False
        reward = 0
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
            log('terminal')
            env.reset()

        grid.draw_background(screen)
        draw_grid(env.grid, screen, CELL_SIZE)
        pygame.display.flip()


if __name__ == '__main__':
    train_loop()
    # human_test_loop()
