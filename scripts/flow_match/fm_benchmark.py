#!/usr/bin/env python
# coding: utf-8

# In[1]:


#@markdown ### **Imports**
# diffusion policy import
from typing import Tuple, Sequence
import numpy as np
import torch
import collections
from tqdm.auto import tqdm

# env import
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import time

from fm import FlowMatchingScheduler
import torch.nn as nn

# from imm.state_nn import RoboIMM as ImmState
# from imm.vision_nn import RoboIMM as ImmVision
import pickle

# In[2]:


#@markdown ### **Environment**
#@markdown Defines a PyMunk-based Push-T environment `PushTEnv`.
#@markdown
#@markdown **Goal**: push the gray T-block into the green area.
#@markdown
#@markdown Adapted from [Implicit Behavior Cloning](https://implicitbc.github.io/)


positive_y_is_up: bool = False
"""Make increasing values of y point upwards.

When True::

    y
    ^
    |      . (3, 3)
    |
    |   . (2, 2)
    |
    +------ > x

When False::

    +------ > x
    |
    |   . (2, 2)
    |
    |      . (3, 3)
    v
    y

"""

def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function wont actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def light_color(color: SpaceDebugColor):
    color = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color

class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        """Draw a pymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import pymunk
        >>> surface = pygame.Surface((10,10))
        >>> space = pymunk.Space()
        >>> options = pymunk.pygame_util.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = pymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordiante system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = pymunk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = pymunk.Body()
        >>> body.position = (0, 0) # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> positive_y_is_up = True
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, round(radius-4), 0)

        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = to_pygame(circle_edge, self.surface)
        line_r = 2 if radius > 20 else 1
        # pygame.draw.lines(self.surface, outline_color.as_int(), False, [p, p2], line_r)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

# env
class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False,
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None, 
            seed=1234
        ):
        self._seed = seed
        self.np_random = np.random.default_rng(seed=self._seed)

        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatiblity
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

        self._setup()

    def reset(self):
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatiblity
        state = self.reset_to_state
        if state is None:
            
            state = np.array([
                self.np_random.integers(50, 450), self.np_random.integers(50, 450),
                self.np_random.integers(100, 400), self.np_random.integers(100, 400),
                self.np_random.uniform(-np.pi, np.pi)
                ])
        self._set_state(state)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold
        terminated = done
        truncated = done

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here dosn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step}
        return info
    
    def _get_state(self):
        state = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))
        return state

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is aleady ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatiblity with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2],
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # Add collision handeling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

class PushTImageEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None,
            damping=None,
            render_size=96,
            seed=1234):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False,
            seed=seed)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None

    def _get_obs(self):
        img = super()._render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }

        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()

        return self.render_cache

# In[3]:


with open('/home/imahajan/diffusion/scripts/flow_match/pusht_data_stats.pkl', 'rb') as f:
    state_data_stats = pickle.load(f)

with open('/home/imahajan/diffusion/scripts/flow_match/vision_data_stats.pkl', 'rb') as f:
    vision_data_stats = pickle.load(f)

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data



def benchmark(model, diffusion_steps=1, mode='state'):
    device = torch.device('cuda')

    # limit enviornment interaction to 350 steps before termination
    max_steps = 350
    if mode == 'state':
        env = PushTEnv(seed=12345)
        desc = "Eval PushTStateEnv"
    elif mode == 'vision':
        env = PushTImageEnv(seed=12345)
        desc = "Eval PushTImageEnv"

    render = False

    episodes = 100
    starts = np.zeros((episodes, 5))
    ends = np.zeros((episodes, 5))
    steps = np.zeros(episodes)
    avgtimes = np.zeros(episodes)

    # save rewards
    final_rewards = np.zeros(episodes)

    with tqdm(total=episodes, desc=desc) as pbar:
        for ep in range(episodes):
            reward = 0.0

            # reset model
            model.reset()
            
            # get first observation
            obs, info = env.reset()
            if render:
                imgs = [env.render(mode='rgb_array')]
            starts[ep] = env._get_state()
            
            # keep a queue of last 2 steps of observations
            obs_deque = collections.deque([obs] * model.obs_horizon, maxlen=model.obs_horizon)

            done = False
            step_idx = 0
            eptimes = []
        
            while not done:
                if mode == 'state':
                    # stack the last obs_horizon (2) number of observations
                    obs_seq = np.stack(obs_deque)
                    # normalize observation
                    nobs = normalize_data(obs_seq, stats=state_data_stats['obs'])
                    # device transfer
                    nobs = torch.from_numpy(nobs).unsqueeze(0).flatten(start_dim=1).to(device, dtype=torch.float32)
                elif mode == 'vision':
                    images = np.stack([x['image'] for x in obs_deque])
                    images = torch.from_numpy(images).unsqueeze(0).to(device, dtype=torch.float32)
                    agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
                    nagent_poses = normalize_data(agent_poses, stats=vision_data_stats['agent_pos'])
                    nagent_poses = torch.from_numpy(nagent_poses).unsqueeze(0).to(device, dtype=torch.float32)
    

                # infer action
                with torch.no_grad():
                    start_time = time.time()
                    if mode == 'state':
                        r = model.sample(shape=(1, model.pred_horizon, model.action_dim), steps=diffusion_steps, global_cond=nobs)
                    elif mode == 'vision':
                        r = model.sample(shape=(1, model.pred_horizon, model.action_dim), image=images, agent_pos=nagent_poses, steps=diffusion_steps)
                    end_time = time.time()
                    eptimes.append(1000 * (end_time - start_time))

                # unnormalize action
                naction = r.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                if mode == 'state':
                    action_pred = unnormalize_data(naction, stats=state_data_stats['action'])
                elif mode == 'vision':
                    action_pred = unnormalize_data(naction, stats=vision_data_stats['action'])

                # only take action_horizon number of actions
                start = model.obs_horizon - 1
                end = start + model.pred_horizon
                action = action_pred[start:end,:]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    if render:
                        imgs.append(env.render(mode='rgb_array'))

                    step_idx += 1
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break
            ends[ep] = env._get_state()
            steps[ep] = step_idx
            avgtimes[ep] = np.mean(eptimes)
            pbar.update(1)
            pbar.set_postfix(avg_reward=sum(final_rewards[:ep+1])/(ep+1))
            final_rewards[ep] = reward
    print(f'Final reward at {diffusion_steps} steps: {sum(final_rewards)/episodes}')

    #pipe to csv epochs, steps, final_reward/episodes
    with open('flow_matching_benchmark.csv', 'a') as f:
        f.write(f'{diffusion_steps}, {sum(final_rewards)/episodes}\n')
    return final_rewards, starts, ends



# Add this wrapper class for your flow matching model
class FlowMatchingWrapper:
    def __init__(self, checkpoint_path, obs_horizon=2, pred_horizon=16, seed=12345):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = 2
        
        # Load checkpoint with weights_only=False to handle the pickle error
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            print(f"First load attempt failed, trying with weights_only=False: {e}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.stats = checkpoint.get('stats', None)
        
        # Create model architecture
        from fm import get_resnet, ConditionalUnet1D, replace_bn_with_gn
        
        # Vision encoder
        self.vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
        
        # Noise prediction network
        vision_feature_dim = 512
        lowdim_obs_dim = 2
        obs_dim = vision_feature_dim + lowdim_obs_dim
        
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=obs_dim*obs_horizon
        )
        
        # Create module dictionary
        self.nets = nn.ModuleDict({
            'vision_encoder': self.vision_encoder,
            'noise_pred_net': self.noise_pred_net
        })
        
        # Load weights
        self.nets.load_state_dict(checkpoint['model_state_dict'])
        self.nets.to(self.device)
        self.nets.eval()
        
        # Initialize flow scheduler
        self.flow_scheduler = FlowMatchingScheduler(num_train_timesteps=100)
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def reset(self):
        # Nothing to reset for flow matching
        pass
    
    def sample(self, shape, image=None, agent_pos=None, steps=16, global_cond=None):
        """Sample from the flow matching model"""
        B, pred_horizon, action_dim = shape
        
        # Process observations
        if global_cond is not None:
            # State-based mode
            obs_cond = global_cond
        else:
            # Vision-based mode
            # Get image features
            image_features = self.nets['vision_encoder'](image.flatten(end_dim=1))
            image_features = image_features.reshape(*image.shape[:2], -1)
            
            # Concatenate vision feature and low-dim obs
            obs_features = torch.cat([image_features, agent_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
        
        # Initialize action from Gaussian noise
        x_t = torch.randn(shape, device=self.device)
        
        # Set up flow matching integration
        self.flow_scheduler.set_timesteps(steps)
        
        # Integrate the ODE: dx/dt = v(x,t)
        for i, t in enumerate(self.flow_scheduler.timesteps):
            # Create batch of current timestep
            t_batch = torch.ones((B,), device=self.device) * t
            
            # Predict velocity at current point
            v_t = self.nets['noise_pred_net'](
                sample=x_t,
                timestep=t_batch,
                global_cond=obs_cond
            )
            
            # Euler integration step
            if i < len(self.flow_scheduler.timesteps) - 1:
                dt = self.flow_scheduler.timesteps[i + 1] - t
                x_t = x_t + dt * v_t
        
        return x_t

# In[17]:
# Add this to your main code to run the flow matching benchmark

print('Flow Matching (40 epochs)')
checkpoint_path = '/home/imahajan/diffusion/checkpoints/flow_matching_epoch_40.pt'
flow_model = FlowMatchingWrapper(checkpoint_path)
bf1 = benchmark(flow_model, 1, mode='vision')
bf2 = benchmark(flow_model, 2, mode='vision')
bf4 = benchmark(flow_model, 4, mode='vision')
bf8 = benchmark(flow_model, 8, mode='vision')
bf16 = benchmark(flow_model, 16, mode='vision')
bf32 = benchmark(flow_model, 32, mode='vision')

print('Flow Matching (60 epochs)')
checkpoint_path = '/home/imahajan/diffusion/checkpoints/flow_matching_epoch_60.pt'
flow_model = FlowMatchingWrapper(checkpoint_path)
bf1 = benchmark(flow_model, 1, mode='vision')
bf2 = benchmark(flow_model, 2, mode='vision')
bf4 = benchmark(flow_model, 4, mode='vision')
bf8 = benchmark(flow_model, 8, mode='vision')
bf16 = benchmark(flow_model, 16, mode='vision')
bf32 = benchmark(flow_model, 32, mode='vision')

print('Flow Matching (80 epochs)')
checkpoint_path = '/home/imahajan/diffusion/checkpoints/flow_matching_epoch_80.pt'
flow_model = FlowMatchingWrapper(checkpoint_path)
bf1 = benchmark(flow_model, 1, mode='vision')
bf2 = benchmark(flow_model, 2, mode='vision')
bf4 = benchmark(flow_model, 4, mode='vision')
bf8 = benchmark(flow_model, 8, mode='vision')
bf16 = benchmark(flow_model, 16, mode='vision')
bf32 = benchmark(flow_model, 32, mode='vision')

print('Flow Matching (100 epochs)')
checkpoint_path = '/home/imahajan/diffusion/checkpoints/flow_matching_epoch_100.pt'
flow_model = FlowMatchingWrapper(checkpoint_path)
bf1 = benchmark(flow_model, 1, mode='vision')
bf2 = benchmark(flow_model, 2, mode='vision')
bf4 = benchmark(flow_model, 4, mode='vision')
bf8 = benchmark(flow_model, 8, mode='vision')
bf16 = benchmark(flow_model, 16, mode='vision')
bf32 = benchmark(flow_model, 32, mode='vision')


