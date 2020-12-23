import pygame
from environment import EnvironmentGrid, CELL_TYPE_BODY, CELL_TYPE_HEAD, CELL_TYPE_FOOD
from typing import Optional
import torch


# FFA1C7
COLOR_FOOD = (255, 161, 199)


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


def draw_gradients(grads: Optional[torch.tensor], surface: pygame.Surface, cell_size: int, position: (int, int)):
    if grads is None:
        return

    rows = len(grads)
    cols = len(grads[0])

    grads = grads.abs()

    min_grad = float(grads.min())
    max_value = float(grads.max()) - min_grad

    def get_color(grad: float) -> (int, int, int):
        value = int(255 * (grad - min_grad) / max_value)
        return value, value, value

    for i in range(rows):
        for j in range(cols):
            grad = grads[i][j]
            color = get_color(grad)
            pygame.draw.rect(
                surface,
                color,
                [
                    j * cell_size + position[0],
                    i * cell_size + position[1],
                    cell_size,
                    cell_size,
                    ]
            )
