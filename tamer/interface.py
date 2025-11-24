import os
import pygame


class Interface:
    """ Pygame interface for training TAMER """

    def __init__(self, action_map, feedback_mode='keyboard'):
        self.action_map = action_map
        self.feedback_mode = feedback_mode
        pygame.init()
        self.font = pygame.font.Font("freesansbold.ttf", 32)

        # set position of pygame window (so it doesn't overlap with gym)
        os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"
        os.environ["SDL_VIDEO_CENTERED"] = "0"

        self.screen = pygame.display.set_mode((200, 100))
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)

    def get_scalar_feedback(self):
        """
        Get human input. 'W' key for positive, 'A' key for negative.
        Returns: scalar reward (1 for positive, -1 for negative)
        """
        reward = 0
        area = None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    area = self.screen.fill((0, 255, 0))
                    reward = 1
                    break
                elif event.key == pygame.K_a:
                    area = self.screen.fill((255, 0, 0))
                    reward = -1
                    break
        pygame.display.update(area)
        return reward
    
    def get_feedback(self):
        reward = 0
        area = None
        for event in pygame.event.get():
            if self.feedback_mode == 'keyboard':
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        area = self.screen.fill((0, 255, 0))
                        reward = 1
                        break
                    elif event.key == pygame.K_a:
                        area = self.screen.fill((255, 0, 0))
                        reward = -1
                        break
            elif self.feedback_mode == 'mouse':
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Scroll up
                        area = self.screen.fill((0, 255, 0))
                        reward = 1
                        break
                    elif event.button == 5:  # Scroll down
                        area = self.screen.fill((255, 0, 0))
                        reward = -1
                        break
        pygame.display.update(area)
        return reward
    

    def show_action(self, action):
        """
        Show agent's action on pygame screen
        Args:
            action: numerical action (for MountainCar environment only currently)
        """
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)
        action_text = str(action)
        if self.action_map is not None and action in self.action_map:
            action_text = self.action_map[action]
        text = self.font.render(action_text, True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        area = self.screen.blit(text, text_rect)
        pygame.display.update(area)

