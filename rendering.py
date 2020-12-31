import pygame
from environment import EnvironmentGrid, CELL_TYPE_BODY, CELL_TYPE_HEAD, CELL_TYPE_FOOD, CELL_TYPE_INTERSECT
from typing import Optional, Tuple
import torch


# FFA1C7
COLOR_FOOD = (255, 161, 199)

cell_color_by_cell_type = dict()
cell_color_by_cell_type[CELL_TYPE_BODY] = (255, 100, 100)
cell_color_by_cell_type[CELL_TYPE_HEAD] = (200, 80, 80)
cell_color_by_cell_type[CELL_TYPE_INTERSECT] = (198, 104, 255)

visible_cells = set(cell_color_by_cell_type.keys())


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    stripped_hex = hex_str[1:] if hex_str[0] == '#' else hex_str
    return (
        int(stripped_hex[:2], 16),
        int(stripped_hex[2:4], 16),
        int(stripped_hex[4:6], 16),
    )


def draw_grid(state: EnvironmentGrid, surface: pygame.Surface, cell_size: int):
    rows = len(state)
    cols = len(state[0])
    for i in range(rows):
        for j in range(cols):
            if state[i][j] in visible_cells:
                color = cell_color_by_cell_type[state[i][j]]
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


COLOR_KEY_NAME = hex_to_rgb('62FF7B')
COLOR_VALUE = hex_to_rgb('FFFFFF')


def draw_stats(surface: pygame.surface, position: (int, int), fps: int, score: int, average_score: float, font: pygame.font.Font):
    text_fps_name = font.render('FPS:', True, COLOR_KEY_NAME)
    text_fps_value = font.render(str(fps), True, COLOR_VALUE)

    text_score_name = font.render('SCORE:', True, COLOR_KEY_NAME)
    text_score_value = font.render(str(score), True, COLOR_VALUE)

    text_avg_score_name = font.render('AVG SCORE:', True, COLOR_KEY_NAME)
    text_avg_score_value = font.render(str(average_score)[:7], True, COLOR_VALUE)

    surface.blit(
        text_fps_name, position
    )
    position = (
        position[0] + text_fps_name.get_rect().width + 10,
        position[1],
    )

    surface.blit(
        text_fps_value, position
    )
    position = (
        position[0] + text_fps_value.get_rect().width + 30,
        position[1],
    )

    surface.blit(
        text_score_name, position
    )
    position = (
        position[0] + text_score_name.get_rect().width + 10,
        position[1],
    )

    surface.blit(
        text_score_value, position
    )
    position = (
        position[0] + text_score_value.get_rect().width + 30,
        position[1],
    )

    surface.blit(
        text_avg_score_name, position
    )
    position = (
        position[0] + text_avg_score_name.get_rect().width + 10,
        position[1],
    )

    surface.blit(
        text_avg_score_value, position
    )
    position = (
        position[0] + text_avg_score_value.get_rect().width + 30,
        position[1],
    )
