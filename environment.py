from typing import Optional, List, DefaultDict, Tuple
from enum import Enum
import numpy as np
import random
from collections import defaultdict

EnvironmentGrid = List[List[int]]
Position = (int, int)
TState = np.ndarray

CELL_TYPE_EMPTY = 0
CELL_TYPE_HEAD = 1
CELL_TYPE_BODY = 2
CELL_TYPE_FOOD = 3
CELL_TYPE_INTERSECT = 4


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
    times_visited_by_location: DefaultDict[Tuple[int, int], int]
    frames_since_food: int

    @staticmethod
    def blank_state(size: (int, int)):
        return [
            [CELL_TYPE_EMPTY] * size[1] for _ in range(size[0])
        ]

    def __init__(self, size: (int, int)):
        self.size = size
        self.num_cells = size[0] * size[1]
        self.state_space = self.num_cells + 1
        self.times_visited_by_location = defaultdict(int)
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

        # Init times visited
        self.times_visited_by_location = defaultdict(int)
        self.times_visited_by_location[self.head_position] = 1
        self.frames_since_food = 0

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
            return -1.0, True

        if (
            self.body_length == (self.size[0] * self.size[1]) - 1 and
            self.grid[r][c] == CELL_TYPE_FOOD
        ):
            return 10.0, True

        return (
            10.0 if self.grid[r][c] == CELL_TYPE_FOOD else 0.0,
            False
        )

    def is_legal_position(self, position: Position) -> bool:
        r, c = position
        return not (r >= self.size[0] or r < 0 or c >= self.size[1] or c < 0 or self.grid[r][c] == CELL_TYPE_BODY)

    def is_forward_direction(self, action: ActionType) -> bool:
        if action not in DIRECTION_BY_ACTION:
            return False

        if self.body_length < 3:
            return True

        if action == ActionType.ACTION_TYPE_RIGHT:
            return self.direction != Direction.LEFT
        elif action == ActionType.ACTION_TYPE_LEFT:
            return self.direction != Direction.RIGHT
        elif action == ActionType.ACTION_TYPE_DOWN:
            return self.direction != Direction.UP
        elif action == ActionType.ACTION_TYPE_UP:
            return self.direction != Direction.DOWN

        return False

    def is_in_bounds(self, position: Position) -> bool:
        r, c = position
        return not (r >= self.size[0] or r < 0 or c >= self.size[1] or c < 0)

    def step(self, action: ActionType) -> (TState, float, bool):
        if self.is_forward_direction(action):
            self.direction = DIRECTION_BY_ACTION[action]

        new_position = move_direction(self.head_position, self.direction)

        reward, is_terminal = self.get_reward(new_position)

        # if we try to move outside the grid, the next state is a blank screen.
        if is_terminal and not self.is_in_bounds(new_position):
            self.grid = Environment.blank_state(self.size)
            return self.grid, reward, is_terminal

        self.grid = self.copy_grid()

        # Figure out if we ate food and need to grow
        ate_food = self.grid[new_position[0]][new_position[1]] == CELL_TYPE_FOOD

        initial_head_position = self.body_head.position
        self.grid[initial_head_position[0]][initial_head_position[1]] = CELL_TYPE_BODY

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

            # Place new food if we haven't won
            if self.body_length != self.size[0] * self.size[1]:
                new_food_position = self.get_new_food_position()
                self.grid[new_food_position[0]][new_food_position[1]] = CELL_TYPE_FOOD
        else:
            # Not growing snake
            prev_tail = self.body_tail

            # Clear cell at previous tail
            r, c = prev_tail.position
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

        # Move the head in the current direction
        previous_head_type = self.grid[new_position[0]][new_position[1]]
        head_type = CELL_TYPE_HEAD if previous_head_type != CELL_TYPE_BODY else CELL_TYPE_INTERSECT
        self.grid[new_position[0]][new_position[1]] = head_type
        self.head_position = new_position

        # Keep track of whether we're going in loops
        if not is_terminal:
            if ate_food:
                self.times_visited_by_location = defaultdict(int)
                self.frames_since_food = 0
            else:
                self.frames_since_food += 1
            self.times_visited_by_location[self.head_position] += 1

        # If the snake is going in loops, terminate the episode
        if (
            self.times_visited_by_location[self.head_position] >= 5 or
            self.frames_since_food >= self.size[0] * self.size[1]
        ):
            is_terminal = True

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

    def print_debug_grid(self):
        char_by_cell_type = dict()
        char_by_cell_type[CELL_TYPE_EMPTY] = ' '
        char_by_cell_type[CELL_TYPE_HEAD] = 'H'
        char_by_cell_type[CELL_TYPE_BODY] = 'B'
        char_by_cell_type[CELL_TYPE_FOOD] = 'F'
        char_by_cell_type[CELL_TYPE_INTERSECT] = 'X'

        for row in self.grid:
            print(''.join(char_by_cell_type[c] for c in row))


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
