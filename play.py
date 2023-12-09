import pygame
import numpy as np


def start(function):
    pygame.init()
    pred = None
    size = width, height = 1000, 800
    font = pygame.font.SysFont("Terminus", 30)

    screen = pygame.display.set_mode(size)

    ROWS, COLS = 28, 28

    OFFSET = 30
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    CELL_SIZE = 20

    handwriting = [[0.0] * COLS for _ in range(ROWS)]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        screen.fill(BLACK)
        click, _, _ = pygame.mouse.get_pressed()
        if click == 1:
            mouse = pygame.mouse.get_pos()
        else:
            mouse = None

        for i in range(ROWS):
            for j in range(COLS):
                rect = pygame.Rect(
                    OFFSET + j * CELL_SIZE, OFFSET + i * CELL_SIZE, CELL_SIZE, CELL_SIZE
                )

                if handwriting[i][j]:
                    channel = 255 - (handwriting[i][j] * 255)
                    channel = int(channel)
                    pygame.draw.rect(screen, (channel, channel, channel), rect)
                else:
                    pygame.draw.rect(screen, WHITE, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)

                if mouse and rect.collidepoint(mouse):
                    handwriting[i][j] = 0.9
                    if i + 1 < ROWS:
                        handwriting[i + 1][j] = 0.9
                    if j + 1 < COLS:
                        handwriting[i][j + 1] = 0.9
                    if i + 1 < ROWS and j + 1 < COLS:
                        handwriting[i + 1][j + 1] = 0.8

        button = pygame.Rect(30, OFFSET + ROWS * CELL_SIZE + 30, 100, 30)
        text_surface = font.render("Reset", True, BLACK)
        text_rect = text_surface.get_rect(
            center=(
                button.x + text_surface.get_width(),
                button.y + text_surface.get_height(),
            )
        )
        pygame.draw.rect(screen, WHITE, button)
        screen.blit(text_surface, text_rect)

        button2 = pygame.Rect(30, OFFSET + ROWS * CELL_SIZE + 100, 100, 30)

        text_surface2 = font.render("Classify", True, BLACK)
        text_rect2 = text_surface2.get_rect(
            center=(
                button2.x + text_surface2.get_width() // 2,
                button2.y + text_surface2.get_height(),
            )
        )
        pygame.draw.rect(screen, WHITE, button2)
        screen.blit(text_surface2, text_rect2)

        if mouse and button.collidepoint(mouse):
            handwriting = [[0.0] * COLS for _ in range(ROWS)]
            pred = None
        if mouse and button2.collidepoint(mouse):
            arr = np.array(handwriting).reshape((784, 1))
            pred = function(arr)
        if pred is not None:
            text = font.render(f"Prediction: {pred}", True, WHITE)
            screen.blit(text, (width - (text.get_width() * 2), 400))
        pygame.display.flip()
