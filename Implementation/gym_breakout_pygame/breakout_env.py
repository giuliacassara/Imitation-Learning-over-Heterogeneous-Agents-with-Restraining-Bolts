# -*- coding: utf-8 -*-

"""
The breakout game is based on CoderDojoSV/beginner-python's tutorial

Luca Iocchi 2017
"""
import math
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Set, Tuple, Dict

import gym
import numpy as np
import pygame
from gym.spaces import Discrete, MultiBinary

Position = Tuple[int, int]

black = [0, 0, 0]
white = [255, 255, 255]
grey = [180, 180, 180]
orange = [180, 100, 20]
red = [180, 0, 0]


class PygameDrawable(ABC):

    @abstractmethod
    def draw_on_screen(self, screen: pygame.Surface):
        """Draw a Pygame object on a given Pygame screen."""


class _AbstractPygameViewer(ABC):

    @abstractmethod
    def reset(self, breakout_state: 'BreakoutState'):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass


class PygameViewer(_AbstractPygameViewer):

    def __init__(self, breakout_state: 'BreakoutState'):
        self.state = breakout_state

        pygame.init()
        pygame.display.set_caption('Breakout')
        self.screen = pygame.display.set_mode([self.state.config.win_width, self.state.config.win_height])
        self.myfont = pygame.font.SysFont("Arial", 30)
        self.drawables = self._init_drawables()  # type: Set[PygameDrawable]

    def reset(self, breakout_state: 'BreakoutState'):
        self.state = breakout_state
        self.drawables = self._init_drawables()

    def _init_drawables(self) -> Set[PygameDrawable]:
        result = set()
        result.add(self.state.ball)
        result.add(self.state.paddle)
        result.add(self.state.brick_grid)
        result.add(self.state.bullet)
        return result

    def render(self, mode="human"):
        self._fill_screen()
        self._draw_score_label()
        self._draw_last_command()
        self._draw_game_objects()

        if mode == "human":
            pygame.display.update()
        elif mode == "rgb_array":
            screen = pygame.surfarray.array3d(self.screen)
            # swap width with height
            return screen.swapaxes(0, 1)

    def _fill_screen(self):
        self.screen.fill(white)

    def _draw_score_label(self):
        score_label = self.myfont.render(str(self.state.score), 100, pygame.color.THECOLORS['black'])
        self.screen.blit(score_label, (50, 10))

    def _draw_last_command(self):
        cmd = self.state.last_command
        s = '%s' % cmd
        count_label = self.myfont.render(s, 100, pygame.color.THECOLORS['brown'])
        self.screen.blit(count_label, (20, 10))

    def _draw_game_objects(self):
        for d in self.drawables:
            d.draw_on_screen(self.screen)

    def close(self):
        pygame.display.quit()
        pygame.quit()


class BreakoutConfiguration(object):

    def __init__(self, brick_rows: int = 3,
                 brick_cols: int = 3,
                 paddle_width: int = 80,
                 paddle_height: int = 10,
                 paddle_speed: int = 10,
                 brick_width: int = 60,
                 brick_height: int = 12,
                 brick_xdistance: int = 20,
                 brick_reward: float = 5.0,
                 step_reward: float = - 0.01,
                 game_over_reward: float = - 10.0,
                 ball_radius: int = 10,
                 resolution_x: int = 20,
                 resolution_y: int = 10,
                 horizon: Optional[int] = None,
                 fire_enabled: bool = False,
                 ball_enabled: bool = True,
                 complex_bump: bool = False,
                 deterministic: bool = True):
        assert brick_cols >= 3, "The number of columns must be at least three."
        assert brick_rows >= 1, "The number of columns must be at least three."
        assert fire_enabled or ball_enabled, "Either fire or ball must be enabled."
        self._brick_rows = brick_rows
        self._brick_cols = brick_cols
        self._paddle_width = paddle_width
        self._paddle_height = paddle_height
        self._paddle_speed = paddle_speed
        self._brick_width = brick_width
        self._brick_height = brick_height
        self._brick_xdistance = brick_xdistance
        self._brick_reward = brick_reward
        self._step_reward = step_reward
        self._game_over_reward = game_over_reward
        self._ball_radius = ball_radius
        self._resolution_x = resolution_x
        self._resolution_y = resolution_y
        self._horizon = horizon if horizon is not None else 300 * (self._brick_cols * self._brick_rows)
        self._complex_bump = complex_bump
        self._deterministic = deterministic
        self._fire_enabled = fire_enabled
        self._ball_enabled = ball_enabled

        self.init_ball_speed_x = 2
        self.init_ball_speed_y = 5
        self.accy = 1.00

    @property
    def win_width(self):
        return int((self._brick_width + self._brick_xdistance) * self._brick_cols + self._brick_xdistance)

    @property
    def win_height(self):
        return 480

    @property
    def n_ball_x(self):
        return self.win_width // self._resolution_x + 1

    @property
    def n_paddle_x(self):
        return self.win_width // self._resolution_x + 1

    @property
    def n_ball_y(self):
        return self.win_height // self._resolution_y + 1

    @property
    def n_ball_dir(self):
        """
        The number of possible ball directions:
        - ball going up (0-5) or down (6-9)
        - ball going left (1,2) straight (0) right (3,4)
        """
        return 10

    @property
    def n_ball_x_speed(self):
        return 5

    @property
    def n_ball_y_speed(self):
        return 2

    @property
    def brick_rows(self):
        return self._brick_rows

    @property
    def brick_cols(self):
        return self._brick_cols

    @property
    def paddle_width(self):
        return self._paddle_width

    @property
    def paddle_height(self):
        return self._paddle_height

    @property
    def paddle_speed(self):
        return self._paddle_speed

    @property
    def brick_width(self):
        return self._brick_width

    @property
    def brick_height(self):
        return self._brick_height

    @property
    def brick_xdistance(self):
        return self._brick_xdistance

    @property
    def brick_reward(self) -> float:
        return self._brick_reward

    @property
    def step_reward(self) -> float:
        return self._step_reward

    @property
    def game_over_reward(self) -> float:
        return self._game_over_reward

    @property
    def ball_radius(self):
        return self._ball_radius

    @property
    def resolution_x(self):
        return self._resolution_x

    @property
    def resolution_y(self):
        return self._resolution_y

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def fire_enabled(self) -> bool:
        return self._fire_enabled

    @property
    def ball_enabled(self) -> bool:
        return self._ball_enabled

    @property
    def complex_bump(self) -> bool:
        return self._complex_bump

    @property
    def deterministic(self) -> bool:
        return self._deterministic


class Command(Enum):
    NOP = 0
    LEFT = 1
    RIGHT = 2
    FIRE = 3

    def __str__(self):
        cmd = Command(self.value)
        if cmd == Command.NOP:
            return "_"
        elif cmd == Command.LEFT:
            return "<"
        elif cmd == Command.RIGHT:
            return ">"
        elif cmd == Command.FIRE:
            return "o"
        else:
            raise ValueError("Shouldn't be here...")


class Brick(PygameDrawable):

    def __init__(self, i: int, j: int, width: int, height: int, xdistance: int,):
        self.i = i
        self.j = j
        self.width = width
        self.height = height
        self.xdistance = xdistance

        self.x = (self.width+self.xdistance)*i+self.xdistance
        self.y = 70+(self.height+8)*j
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def draw_on_screen(self, screen):
        pygame.draw.rect(screen, grey, self.rect, 0)


class BrickGrid(PygameDrawable):

    def __init__(self, brick_cols: int,
                 brick_rows: int,
                 brick_width: int,
                 brick_height: int,
                 brick_xdistance: int):
        self.brick_cols = brick_cols
        self.brick_rows = brick_rows
        self.brick_width = brick_width
        self.brick_height = brick_height
        self.brick_xdistance = brick_xdistance

        self.bricks = {}  # type: Dict[Tuple[int, int], Brick]
        self.bricksgrid = np.zeros((self.brick_cols, self.brick_rows))
        self._init_bricks()

    def _init_bricks(self):
        for i in range(0, self.brick_cols):
            for j in range(0, self.brick_rows):
                temp = Brick(i, j, self.brick_width, self.brick_height, self.brick_xdistance)
                self.bricks[(i, j)] = temp
                self.bricksgrid[i][j] = 1

    def draw_on_screen(self, screen: pygame.Surface):
        for b in self.bricks.values():
            b.draw_on_screen(screen)

    def remove_brick_at_position(self, position: Position):
        self.bricks.pop(position)
        self.bricksgrid[position[0], position[1]] = 0

    def is_empty(self):
        return len(self.bricks) == 0


class Ball(PygameDrawable):

    def __init__(self, breakout_config: BreakoutConfiguration):
        self.config = breakout_config

        if breakout_config.ball_enabled:
            _initial_ball_x = self.config.win_width // 2
            _initial_ball_y = self.config.win_height - 100 - self.config._ball_radius
            self.x = _initial_ball_x
            self.y = _initial_ball_y
            self.speed_x = self.config.init_ball_speed_x
            self.speed_y = self.config.init_ball_speed_y
            self._radius = self.config.ball_radius
        else:
            self.x = 0
            self.y = 0
            self.speed_x = 0.0
            self.speed_y = 0.0
            self._radius = 0

    @property
    def radius(self):
        return self._radius

    @property
    def speed_x_norm(self) -> int:
        if self.speed_x < -2.5:
            return 0
        elif -2.5 <= self.speed_x < 0:
            return 1
        elif self.speed_x == 0:
            return 2
        elif 0 < self.speed_x < 2.5:
            return 3
        elif 2.5 <= self.speed_x:
            return 4
        else:
            raise ValueError("Speed x not recognized.")

    @property
    def speed_y_norm(self) -> int:
        if self.speed_y <= 0:
            return 0
        else:
            return 1

    @property
    def dir(self):
        ball_dir = 0
        if self.speed_y > 0:  # down
            ball_dir += 5
        if self.speed_x < -2.5:  # quick-left
            ball_dir += 1
        elif self.speed_x < 0:  # left
            ball_dir += 2
        elif self.speed_x > 2.5:  # quick-right
            ball_dir += 3
        elif self.speed_x > 0:  # right
            ball_dir += 4
        return ball_dir

    def draw_on_screen(self, screen: pygame.Surface):
        pygame.draw.circle(screen, orange, [int(self.x), int(self.y)], self.radius, 0)

    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y


class Paddle(PygameDrawable):

    def __init__(self, breakout_config: BreakoutConfiguration):
        self.config = breakout_config

        _initial_paddle_x = self.config.win_width // 2
        _initial_paddle_y = self.config.win_height - 20
        self.x = _initial_paddle_x
        self.y = _initial_paddle_y

    @property
    def width(self):
        return self.config._paddle_width

    @property
    def height(self):
        return self.config._paddle_height

    @property
    def speed(self):
        return self.config._paddle_speed

    def draw_on_screen(self, screen: pygame.Surface):
        pygame.draw.rect(screen, grey, [self.x, self.y, self.width, self.height], 0)

    def update(self, command: Command):
        if command == Command.LEFT:
            self.x -= self.speed
        elif command == Command.RIGHT:
            self.x += self.speed
        elif command == Command.NOP:
            pass
        elif command == Command.FIRE:
            pass
        else:
            raise Exception("Command not recognized.")

        if self.x < 0:
            self.x = 0
        if self.x > self.config.win_width - self.width:
            self.x = self.config.win_width - self.width


class Bullet(PygameDrawable):

    def __init__(self, breakout_config: BreakoutConfiguration):
        self.config = breakout_config

        self.x = 0.0
        self.y = 0.0
        self.speed_y = 0.0

    @property
    def in_movement(self):
        return self.speed_y < 0.0

    @property
    def width(self):
        return 5

    @property
    def height(self):
        return 5

    def update(self):
        self.y += self.speed_y
        if self.y < 5:
            self.reset()

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.speed_y = 0.0

    def draw_on_screen(self, screen: pygame.Surface):
        if self.speed_y < 0:
            pygame.draw.rect(screen, red, [self.x, self.y, self.width, self.height], 0)


class BreakoutState(object):

    def __init__(self, breakout_configuration: BreakoutConfiguration):
        self.config = breakout_configuration

        self.ball = Ball(self.config)
        self.paddle = Paddle(self.config)
        self.brick_grid = BrickGrid(self.config.brick_cols,
                                    self.config.brick_rows,
                                    self.config.brick_width,
                                    self.config.brick_height,
                                    self.config.brick_xdistance)
        
        self.bullet = Bullet(self.config)

        self.last_command = Command.NOP  # type: Command
        self.score = 0
        self._steps = 0

    def reset(self) -> 'BreakoutState':
        return BreakoutState(self.config)

    def update(self, command: Command):
        self.paddle.update(command)
        self.ball.update()
        self.bullet.update()
        self.last_command = str(command)

    def remove_brick_at_position(self, position: Position):
        self.brick_grid.remove_brick_at_position(position)

    def to_dict(self) -> Dict:
        """Extract the state observation based on the game configuration."""

        ball_x = int(self.ball.x) // self.config.resolution_x
        ball_y = int(self.ball.y) // self.config.resolution_y
        ball_x_speed = self.ball.speed_x_norm
        ball_y_speed = self.ball.speed_y_norm
        paddle_x = int(self.paddle.x) // self.config.resolution_x
        bricks_matrix = self.brick_grid.bricksgrid

        return {
            "paddle_x": paddle_x,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "ball_x_speed": ball_y_speed,
            "ball_y_speed": ball_x_speed,
            "bricks_matrix": bricks_matrix,
        }

    def step(self, command: Command) -> int:
        """
        Check collisions and update the state of the game accordingly.

        :return: the reward resulting from this step.
        """
        reward = 0
        self._steps += 1
        self.update(command)

        ball = self.ball
        paddle = self.paddle
        bullet = self.bullet
        brick_grid = self.brick_grid

        ball_rect = pygame.Rect(ball.x - ball.radius,
                                ball.y - ball.radius,
                                ball.radius * 2,
                                ball.radius * 2)
        paddle_rect = pygame.Rect(paddle.x, paddle.y, paddle.width, paddle.height)
        bullet_rect = pygame.Rect(bullet.x, bullet.y, bullet.width, bullet.height)

        # for screen border
        if ball.y < ball.radius:
            ball.y = ball.radius
            ball.speed_y = - ball.speed_y
            if np.isclose(ball.speed_x, 0.0):
                ball.speed_x = 1.0 * random.choice([-1.0, 1.0])
        if ball.x < ball.radius:
            ball.x = ball.radius
            ball.speed_x = - ball.speed_x
        if ball.x > self.config.win_width - ball.radius:
            ball.x = self.config.win_width - ball.radius
            ball.speed_x = - ball.speed_x

        # for paddle
        if ball_rect.colliderect(paddle_rect):
            if self.config.complex_bump:
                dbp = math.fabs(ball.x - (paddle.x + paddle.width / 2))
                if dbp < 20:
                    # print 'straight'
                    if (ball.speed_x < -5):
                        ball.speed_x += 2
                    elif (ball.speed_x > 5):
                        ball.speed_x -= 2
                    elif (ball.speed_x <= -0.5):
                        ball.speed_x += 0.5
                    elif (ball.speed_x >= 0.5):
                        ball.speed_x -= 0.5

                dbp = math.fabs(ball.x - (paddle.x + 0))
                if dbp < 10:
                    # print 'left'
                    ball.speed_x = -abs(ball.speed_x) - 1
                dbp = math.fabs(ball.x - (paddle.x + paddle.width))
                if dbp < 10:
                    # print 'right'
                    ball.speed_x = abs(ball.speed_x) + 1

            else:
                dbp = math.fabs(ball.x - (paddle.x + paddle.width / 2))
                if dbp < 20:
                    # print 'straight'
                    if (ball.speed_x != 0):
                        ball.speed_x = 2 * abs(ball.speed_x) / ball.speed_x
                dbp = math.fabs(ball.x - (paddle.x + 0))
                if dbp < 20:
                    # print 'left'
                    ball.speed_x = -5
                    RandomEventGenerator.perturbate_ball_speed_after_paddle_hit(self)
                dbp = math.fabs(ball.x - (paddle.x + paddle.width))
                if dbp < 20:
                    # print 'right'
                    ball.speed_x = 5
                    RandomEventGenerator.perturbate_ball_speed_after_paddle_hit(self)

            ball.speed_y = - abs(ball.speed_y)

        for brick in brick_grid.bricks.values():
            if brick.rect.colliderect(ball_rect):
                self.score += self.config.brick_reward
                self.remove_brick_at_position((brick.i, brick.j))
                ball.speed_y = - ball.speed_y
                reward += self.config.brick_reward
                break

        if command == Command.FIRE:  # fire
            if not bullet.in_movement:
                bullet.x = paddle.x + paddle.width / 2
                bullet.y = paddle.y
                bullet.speed_y = -10

        # firing
        if bullet.y < 5:
            # reset
            bullet.reset()

        for brick in brick_grid.bricks.values():
            if brick.rect.colliderect(bullet_rect):
                self.remove_brick_at_position((brick.i, brick.j))
                reward += self.config.brick_reward
                self.score += self.config.brick_reward
                self.bullet.reset()
                break

        reward += self.config.step_reward

        # ball out
        reward += self.config.game_over_reward if self.ball.y > self.config.win_height - self.ball.radius else 0
        # time out
        reward += self.config.game_over_reward if self._steps > self.config.horizon else 0.0

        return reward

    def is_finished(self):
        end1 = self.ball.y > self.config.win_height - self.ball.radius
        end2 = self.brick_grid.is_empty()
        end3 = self._steps > self.config.horizon
        return end1 or end2 or end3


class RandomEventGenerator:

    @classmethod
    def perturbate_initial_ball_speed(cls, state: BreakoutState):
        if not state.config.deterministic:
            ran = random.uniform(0.75, 1.5)
            state.ball.speed_x *= ran
            # print(print("random ball_speed_x = %.2f" %self.ball_speed_x)

    @classmethod
    def perturbate_ball_speed_after_brick_hit(cls, state: BreakoutState):
        if not state.config.deterministic:
            ran = random.uniform(0.0, 1.0)
            if ran < 0.5:
                state.ball.speed_x *= -1
            # print("random ball_speed_x = %.2f" %self.ball_speed_x)

    @classmethod
    def perturbate_ball_speed_after_paddle_hit(cls, state: BreakoutState):
        if not state.config.deterministic:
            ran = random.uniform(0.0, 1.0)
            if ran < 0.1:
                state.ball.speed_x *= 0.75
            elif ran > 0.9:
                state.ball.speed_x *= 1.5
            sign = state.ball.speed_x / abs(state.ball.speed_x)
            state.ball.speed_x = min(state.ball.speed_x, 6) * sign
            state.ball.speed_x = max(state.ball.speed_x, 0.5) * sign
            # print("random ball_speed_x = %.2f" %self.ball_speed_x)


class Breakout(gym.Env, ABC):
    """A generic Breakout env. The feature space must be define in subclasses."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, breakout_config: Optional[BreakoutConfiguration] = None):

        self.config = BreakoutConfiguration() if breakout_config is None else breakout_config
        self.state = BreakoutState(self.config)
        self.viewer = None  # type: Optional[PygameViewer]

        self.action_space = Discrete(len(Command) if self.config.fire_enabled else len(Command) - 1)

        self._paddle_x_space = Discrete(self.config.n_paddle_x)
        self._ball_x_space = Discrete(self.config.n_ball_x)
        self._ball_y_space = Discrete(self.config.n_ball_y)
        self._ball_x_speed_space = Discrete(self.config.n_ball_x_speed)
        self._ball_y_speed_space = Discrete(self.config.n_ball_y_speed)
        self._ball_dir_space = Discrete(self.config.n_ball_dir)
        self._bricks_matrix_space = MultiBinary((self.config._brick_rows, self.config._brick_cols))

    def step(self, action: int):
        command = Command(action)
        reward = self.state.step(command)
        obs = self.observe(self.state)
        is_finished = self.state.is_finished()
        info = {}
        return obs, reward, is_finished, info

    def reset(self):
        self.state = BreakoutState(self.config)
        if self.viewer is not None:
            self.viewer.reset(self.state)
        return self.observe(self.state)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = PygameViewer(self.state)

        return self.viewer.render(mode=mode)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    @abstractmethod
    def observe(self, state: BreakoutState):
        """
        Extract observation from the state of the game.
        :param state: the state of the game
        :return: an instance of a gym.Space
        """

    def play(self):
        self.reset()
        self.render()
        quitted = False
        while not quitted:
            pygame.time.wait(10)
            cmd = 0
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                    quitted = True

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_LEFT]:
                cmd = 1
            elif pressed[pygame.K_RIGHT]:
                cmd = 2
            elif pressed[pygame.K_SPACE]:
                cmd = 3

            _, _, done, _ = self.step(cmd)
            if done: self.reset()
            self.render()
