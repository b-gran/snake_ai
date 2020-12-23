from typing import Optional, List
from enum import Enum
import numpy as np
import random

EnvironmentGrid = List[List[int]]
Position = (int, int)
TState = np.ndarray

CELL_TYPE_EMPTY = 0
CELL_TYPE_HEAD = 1
CELL_TYPE_BODY = 2
CELL_TYPE_FOOD = 3


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class ActionType(Enum):
    ACTION_TYPE_NOTHING = 0
    ACTION_TYPE_LEFT = 1
    ACTION_TYPE_UP = 2
    ACTION_TYPE_RIGHT = 3
    ACTION_TYPE_DOWN = 4


DIRECTION_BY_ACTION = {
    ActionType.ACTION_TYPE_LEFT: Direction.LEFT,
    ActionType.ACTION_TYPE_RIGHT: Direction.RIGHT,
    ActionType.ACTION_TYPE_UP: Direction.UP,
    ActionType.ACTION_TYPE_DOWN: Direction.DOWN,
}


class BodyNode:
    position: Position
    next: Optional['BodyNode']
    prev: Optional['BodyNode']

    def __init__(self, position: Position):
        self.position = position
        self.next = None
        self.prev = None


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


def add_position(a: Position, b: Position) -> Position:
    return a[0] + b[0], a[1] + b[1]
