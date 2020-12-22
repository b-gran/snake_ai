import pygame

Color = (int, int, int)

# 283647
BACKGROUND_DARK = (40, 54, 71)

# 344E69
BACKGROUND_LIGHT = (52, 78, 105)


class Grid:
    def __init__(
        self, rows: int, columns: int, cell_size: int, background_light: Color = BACKGROUND_LIGHT,
        background_dark: Color = BACKGROUND_DARK
    ):
        self.rows = rows
        self.columns = columns
        self.cell_size = cell_size
        self.width = rows * cell_size
        self.height = columns * cell_size
        self.offset = (self.columns - 1) % 2

    def draw_background(self, surface: pygame.Surface):
        for i in range(self.rows):
            for j in range(self.columns):
                cell_color = BACKGROUND_DARK if (i * self.rows + j - i * self.offset) % 2 == 0 else BACKGROUND_LIGHT
                pygame.draw.rect(
                    surface,
                    cell_color,
                    [j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size]
                )
