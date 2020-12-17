import pygame
from pygame.locals import *
import math
from enum import Enum, auto
from typing import Optional, Iterable, List, Deque
import random
import collections

CELL_SIZE = 60

# 283647
BACKGROUND_DARK = (40, 54, 71)

# 344E69
BACKGROUND_LIGHT = (52, 78, 105)

# FFA1C7
COLOR_FOOD = (255, 161, 199)

Position = (int, int)

class Grid:
	def __init__(self, width: int, height: int):
		self.width = width
		self.height = height
		self.rows = math.floor(self.height / CELL_SIZE)
		self.cols = math.floor(self.width / CELL_SIZE)

	def draw_background(self, surface: pygame.Surface):
		for i in range(self.rows):
			for j in range(self.cols):
				cell_color = BACKGROUND_DARK if (i * self.rows + j) % 2 == 0 else BACKGROUND_LIGHT
				pygame.draw.rect(surface, cell_color, [j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE])


class Direction(Enum):
	UP = auto()
	RIGHT = auto()
	DOWN = auto()
	LEFT = auto()


class BodyNode:
	position: Position
	next: Optional['BodyNode']
	prev: Optional['BodyNode']

	def __init__(self, position: Position):
		self.position = position
		self.next = None
		self.prev = None


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


class CellType(Enum):
	BODY = auto()
	FOOD = auto()
	EMPTY = auto()


class Player:
	grid: Grid
	body_tail: BodyNode
	body_head: BodyNode
	input_buffer: Deque[Direction]

	def __init__(self, grid: Grid):
		self.grid = grid

		self.max_length = 1
		self.length = 1

		self.position = (int(grid.rows / 2), int(grid.cols / 2))
		self.direction = Direction.RIGHT

		self.body_state = [[CellType.EMPTY] * grid.cols for _ in range(grid.rows)]
		self.body_state[self.position[0]][self.position[1]] = CellType.BODY

		self.body_head = BodyNode(self.position)
		self.body_tail = self.body_head

		self.input_buffer = collections.deque(maxlen=2)

	def can_move(self, direction: Direction) -> bool:
		r, c = move_direction(self.position, direction)

		if r >= self.grid.rows or r < 0 or c >= self.grid.cols or c < 0 or self.body_state[r][c] == CellType.BODY:
			return False

		return True

	def move(self, direction: Direction):
		new_position = move_direction(self.position, direction)

		self.position = new_position
		self.body_state[new_position[0]][new_position[1]] = CellType.BODY

		# Don't need to grow the snake
		if self.max_length == self.length:
			prev_tail = self.body_tail

			# Clear cell at previous tail
			r, c = prev_tail.position
			self.body_state[r][c] = CellType.EMPTY

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
		else:
			# Snake should grow
			self.length += 1

			# Create new head
			new_head = BodyNode(new_position)

			# Add in front of current head
			self.body_head.next = new_head
			new_head.prev = self.body_head

			# Update head
			self.body_head = new_head

	def enqueue_input(self, direction):
		self.input_buffer.append(direction)

	def pop_input(self):
		if len(self.input_buffer) > 0:
			return self.input_buffer.popleft()

		return None

	def try_set_direction(self, direction):
		if self.direction == Direction.UP and direction == Direction.DOWN:
			return
		if self.direction == Direction.DOWN and direction == Direction.UP:
			return
		if self.direction == Direction.LEFT and direction == Direction.RIGHT:
			return
		if self.direction == Direction.RIGHT and direction == Direction.LEFT:
			return

		self.direction = direction

	def body_iter(self) -> Iterable[BodyNode]:
		curr = self.body_tail
		while curr:
			yield curr
			curr = curr.next

	def draw(self, surface: pygame.Surface):
		for n in self.body_iter():
			pygame.draw.rect(
				surface,
				(255, 100, 100),
				[
					n.position[1] * CELL_SIZE,
					n.position[0] * CELL_SIZE,

					# TODO: put CELL_SIZE on grid?
					CELL_SIZE,
					CELL_SIZE,
				]
			)

	def is_on_food(self, food):
		return self.position[0] == food.position[0] and self.position[1] == food.position[1]

	def get_new_food_position(self, food):
		valid_positions = []
		for r in range(len(self.body_state)):
			for c in range(len(self.body_state[r])):
				if self.body_state[r][c] != CellType.BODY and not (
					r == food.position[0] and
					c == food.position[1]
				):
					valid_positions.append((r, c))

		if len(valid_positions) == 0:
			raise Exception('No valid positions on grid')

		return random.choice(valid_positions)


class Food:
	position: Position

	def __init__(self, position: Position):
		self.position = position

	def draw(self, surface):
		r, c = self.position
		# circle(surface, color, center, radius)
		pygame.draw.circle(surface, COLOR_FOOD, [c*CELL_SIZE + CELL_SIZE/2, r*CELL_SIZE + CELL_SIZE/2], int(CELL_SIZE/3))


class Events(Enum):
	MOVEMENT_TICK = pygame.USEREVENT


def main():
	pygame.init()

	# TODO: level class
	pygame.time.set_timer(Events.MOVEMENT_TICK.value, 100)

	grid = Grid(800, 800)
	player = Player(grid)
	food = Food((0, 0))

	clock = pygame.time.Clock()

	screen = pygame.display.set_mode((grid.width, grid.height))
	pygame.display.set_caption('Snake')

	# Event loop
	while True:
		clock.tick(60)

		for event in pygame.event.get():
			if event.type == QUIT:
				return

			if event.type == KEYDOWN:
				if event.key == K_UP:
					player.enqueue_input(Direction.UP)
				elif event.key == K_RIGHT:
					player.enqueue_input(Direction.RIGHT)
				elif event.key == K_DOWN:
					player.enqueue_input(Direction.DOWN)
				elif event.key == K_LEFT:
					player.enqueue_input(Direction.LEFT)

			if event.type == Events.MOVEMENT_TICK.value:
				new_direction = player.pop_input()
				if new_direction:
					player.try_set_direction(new_direction)

				if player.can_move(player.direction):
					player.move(player.direction)
				else:
					print('Would die')

			# print(event)

		if player.is_on_food(food):
			player.max_length += 1
			food.position = player.get_new_food_position(food)

		screen.fill((0, 0, 0))
		grid.draw_background(screen)
		player.draw(screen)
		food.draw(screen)

		pygame.display.flip()


if __name__ == '__main__':
	main()
