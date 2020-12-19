import pygame
from typing import Callable


class Button(pygame.sprite.Sprite):
	def __init__(
		self, x: int, y: int, width: int, height: int, surface: pygame.Surface, surface_hover: pygame.Surface,
		surface_click: pygame.Surface, text: str, text_color: (int, int, int), font: pygame.font.Font,
		on_click: Callable[[], None]
	):
		super().__init__()

		# Scale the images to the desired size (doesn't modify the originals).
		self.surface = pygame.transform.scale(surface, (width, height))
		self.surface_hover = pygame.transform.scale(surface_hover, (width, height))
		self.surface_click = pygame.transform.scale(surface_click, (width, height))

		self.image = self.surface

		self.rect = self.surface.get_rect(topleft=(x, y))

		image_center = self.surface.get_rect().center
		text_surface = font.render(text, True, text_color)
		text_rect = text_surface.get_rect(center=image_center)

		# Blit the text onto the images.
		for image in (self.surface, self.surface_hover, self.surface_click):
			image.blit(text_surface, text_rect)

		self.on_click = on_click
		self.is_clicked = False

	def handle_event(self, event: pygame.event.Event):
		if event.type == pygame.MOUSEBUTTONDOWN:
			if self.rect.collidepoint(event.pos):
				self.image = self.surface_click
				self.is_clicked = True
		elif event.type == pygame.MOUSEBUTTONUP:
			# If the rect collides with the mouse pos.
			if self.rect.collidepoint(event.pos) and self.is_clicked:
				self.on_click()  # Call the function.
				self.image = self.surface_hover
			self.is_clicked = False
		elif event.type == pygame.MOUSEMOTION:
			collided = self.rect.collidepoint(event.pos)
			if collided and not self.is_clicked:
				self.image = self.surface_hover
			elif not collided:
				self.image = self.surface
