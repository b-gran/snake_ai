import pygame
from typing import Callable, List, Union, Iterable
from enum import Enum, auto

RectOrSprite = Union[pygame.Rect, pygame.sprite.Sprite]


def get_rect(rect_or_sprite: RectOrSprite) -> pygame.Rect:
    return rect_or_sprite if rect_or_sprite is pygame.Rect else rect_or_sprite.rect


class LayoutType(Enum):
    CENTER = auto()


class LayoutDirection(Enum):
    VERTICAL = auto()


class LayoutLocation(Enum):
    CENTER = auto()


class Layout:
    children: List[RectOrSprite]

    def __init__(
        self,
        hlayout=LayoutType.CENTER,
        vlayout=LayoutType.CENTER,
        vpadding_inner=0,
        direction=LayoutDirection.VERTICAL
    ):
        self.children = []
        self.hlayout = hlayout
        self.vlayout = vlayout
        self.vpadding_inner = vpadding_inner
        self.direction = direction

    def add(self, child: RectOrSprite):
        self.children.append(child)

    def add_all(self, children: Iterable[RectOrSprite]):
        self.children.extend(children)

    def update(self, container: pygame.Rect, location=(LayoutLocation, LayoutLocation)):
        bounding_rect = self.get_bounding_rect()
        if location[0] == LayoutLocation.CENTER:
            bounding_rect.centerx = container.width / 2
        if location[1] == LayoutLocation.CENTER:
            bounding_rect.centery = container.height / 2

        self.update_absolute((bounding_rect.left, bounding_rect.top))

    def update_absolute(self, top_left=(int, int)):
        child_rects = [get_rect(child) for child in self.children]
        max_width = max(child.width for child in child_rects)
        max_height = max(child.height for child in child_rects)

        if self.direction == LayoutDirection.VERTICAL:
            current_top = top_left[1]
            for child in child_rects:
                if self.hlayout == LayoutType.CENTER:
                    child.left = top_left[0] + (max_width - child.width) / 2
                child.top = current_top
                current_top += child.height + self.vpadding_inner

    def get_bounding_rect(self) -> pygame.Rect:
        child_rects = [get_rect(child) for child in self.children]
        max_width = max(child.width for child in child_rects)
        max_height = max(child.height for child in child_rects)

        rect = pygame.Rect(0, 0, 0, 0)

        if self.direction == LayoutDirection.VERTICAL:
            rect.width = max_width
            rect.height = sum(child.height for child in child_rects) + (len(self.children) - 1) * self.vpadding_inner

        return rect
