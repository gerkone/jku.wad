"""
Adapted from https://github.com/tencent-ailab/Arena

Arena: a toolkit for Multi-Agent Reinforcement Learning https://arxiv.org/abs/1907.09467
All credits to authors.
"""

from typing import Callable, Optional, Sequence, List, Union, Tuple

from copy import deepcopy
from collections import deque
from portpicker import pick_unused_port
import numpy as np
import torch
from collections import defaultdict
from warnings import warn
import copy

import vizdoom as vzd
from gymnasium import Env
from torchvision import transforms
from gymnasium.spaces import Box, MultiDiscrete, Discrete

from doom_arena.parallel import ParallelEnv
from doom_arena.utils import (
    get_doom_buttons,
    get_screen_shape,
    CHANNELS_FORMAT,
    to_tensor,
    resize,
    minmax,
    cat_dict,
)
from doom_arena.reward import VizDoomReward
from doom_arena.player import (
    ObsBuffer,
    PlayerHostConfig,
    PlayerJoinConfig,
    PlayerConfig,
    player_setup,
    player_host_setup,
    player_join_setup,
    mp_game_setup,
)

# TODO what does this do?
VIZDOOM_ACTIONS = [
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # 0 move fast forward
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 fire
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # 2 move left
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 3 move right
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # 4 turn left
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 5 turn right
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 20],  # 6 turn left 20 degree and move forward
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 20],  # 7 turn right 20 degree and move forward
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 8 move forward
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 9 turn 180
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 10 move left
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 11 move right
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 12 turn left
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 13 turn right
]


class PlayerEnv(Env):
    """ViZDoom per player environment."""

    def __init__(
        self,
        cfg: PlayerConfig,
        discrete7: bool = True,
        frame_skip: int = 1,
        seed: int = 1337,
    ):
        self.cfg = cfg
        self.discrete7 = discrete7
        self.transform = cfg.transform
        self.record = False
        self.replay = None
        self.is_spectator = cfg.player_mode == vzd.Mode.SPECTATOR
        self.frame_skip = frame_skip
        # self.seed(seed)
        self.doom_seed = seed
        self.game = None

        self.observation_space = Box(
            low=0,
            high=255,
            dtype=np.uint8,
            shape=get_screen_shape(
                cfg.screen_format,
                cfg.screen_resolution,
                labels=cfg.use_labels,
                depth=cfg.use_depth,
                automap=cfg.use_automap,
            ),
        )

        self._buttons = get_doom_buttons(cfg)
        if discrete7:
            # simplified
            self.action_space = Discrete(len(self._buttons) + 1)  # NOTE: +1 for noop
        else:
            # TODO what does this do?
            # actual button space
            self.action_space = MultiDiscrete([2] * len(self._buttons))

        self._state = None
        self._game_var_list = None
        self._game_vars = {}
        self._game_vars_pre = {}

        self.history = deque(maxlen=60)
        self.frame_stack = deque(maxlen=cfg.n_stack_frames)

    def repeat_action(self, action):
        cnt = 1
        while len(self.history) > cnt + 1 and self.history[-(cnt + 1)] == action:
            cnt += 1
        return cnt

    def adjust_action(self, action):
        # TODO what does this do?
        self.history.append(action)
        is_attacking = VIZDOOM_ACTIONS[1] in list(self.history)[-3:]
        if action in [VIZDOOM_ACTIONS[4], VIZDOOM_ACTIONS[5]]:
            if self.repeat_action(action) > 3:
                action = (
                    VIZDOOM_ACTIONS[6]
                    if action == VIZDOOM_ACTIONS[4]
                    else VIZDOOM_ACTIONS[7]
                )
        if action in [VIZDOOM_ACTIONS[0], VIZDOOM_ACTIONS[2], VIZDOOM_ACTIONS[3]]:
            if self.repeat_action(action) % 16 == 0:
                action = VIZDOOM_ACTIONS[9]
        if action in [
            VIZDOOM_ACTIONS[0],
            VIZDOOM_ACTIONS[2],
            VIZDOOM_ACTIONS[3],
            VIZDOOM_ACTIONS[4],
            VIZDOOM_ACTIONS[5],
        ]:
            if is_attacking:
                if action == VIZDOOM_ACTIONS[0]:
                    action = VIZDOOM_ACTIONS[8]
                if action == VIZDOOM_ACTIONS[2]:
                    action = VIZDOOM_ACTIONS[10]
                if action == VIZDOOM_ACTIONS[3]:
                    action = VIZDOOM_ACTIONS[11]
                if action == VIZDOOM_ACTIONS[4]:
                    action = VIZDOOM_ACTIONS[12]
                if action == VIZDOOM_ACTIONS[5]:
                    action = VIZDOOM_ACTIONS[13]
        if self.game.is_player_dead():
            self.history.clear()
        # convert to c++ type
        action = list(action)
        return action

    def reset(self):
        self._init_game()
        if self.cfg.num_bots > 0:
            self._add_bot()

        self.game.new_episode()
        self._state, obs, _ = self._grab()

        self._record_replay(obs, reset=True)
        self._update_game_vars(reset=True)
        obs = self._update_frame_stack(obs, reset=True)

        # apply (frame) transforms
        if self.transform is not None:
            obs = self.transform(obs)
        return obs

    def step(self, action):
        if isinstance(action, np.ndarray) or isinstance(action, torch.Tensor):
            action = action.tolist()
        if self.discrete7:
            num_configured_buttons = len(self._buttons)
            processed_action = [0] * num_configured_buttons
            if (
                action > 0
            ):  # action is 1-indexed from Discrete(num_buttons+1) space; 0 is no-op.
                # Valid button actions are 1 to num_configured_buttons.
                button_index = action - 1
                if (
                    button_index < num_configured_buttons
                ):  # Check if button_index is valid
                    processed_action[button_index] = 1
                # If action > num_configured_buttons (i.e., action > self.action_space.n -1),
                # it's an invalid action from the space, results in a no-op.
                # This should ideally not happen if sampling correctly from the space.
            action = processed_action  # Use the processed list for ViZDoom
        # action = self.adjust_action(action)
        # vizdoom step
        rwd = self.game.make_action(action, tics=self.frame_skip)
        self._state, obs, done = self._grab()

        self._record_replay(obs)
        self._update_game_vars()
        obs = self._update_frame_stack(obs)
        # apply (frame) transforms
        if self.transform is not None:
            obs = self.transform(obs)
        return obs, rwd, done, {}

    def close(self):
        if self.game:
            self.game.close()

    def _init_game(self):
        self.close()
        game = vzd.DoomGame()
        game = player_setup(game, self.cfg)
        game = mp_game_setup(game, self.cfg.bot_skill)
        if self.cfg.host_cfg is not None:
            game = player_host_setup(game, self.cfg.host_cfg)
        elif self.cfg.join_cfg is not None:
            game = player_join_setup(game, self.cfg.join_cfg)
        else:
            raise ValueError("neither host nor join, error!")
        game.set_window_visible(False)  # NOTE: for devices without screen
        game.set_seed(self.doom_seed)
        game.init()
        self.game = game
        # game variables
        self._game_var_list = self.game.get_available_game_variables()
        self._update_game_vars(reset=True)
        # active buffers
        self._buffers = ["screen"]
        if self.game.is_labels_buffer_enabled():
            self._buffers.append("labels")
        if self.game.is_depth_buffer_enabled():
            self._buffers.append("depth")
        if self.game.is_automap_buffer_enabled():
            self._buffers.append("automap")

    def _grab(self):
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        # game over, all obs to white
        if done and state is None:
            screen_channels = CHANNELS_FORMAT[self.cfg.screen_format]
            obs = {}
            for k in self._buffers:
                shape = get_screen_shape(
                    self.cfg.screen_format,
                    self.cfg.screen_resolution,
                    labels=(k == "labels"),
                    depth=(k == "depth"),
                    automap=(k == "automap"),
                )
                shape = list(shape)
                if k != "screen":
                    shape[0] = shape[0] - screen_channels  # remove screen channels
                obs[k] = np.ones(shape, dtype=np.uint8) * 255
        else:
            screen = state.screen_buffer
            # add channels to grayscale
            screen = screen[None] if screen.ndim == 2 else screen
            obs = {"screen": screen}
            if state.labels_buffer is not None:
                obs["labels"] = state.labels_buffer[None]
            if state.depth_buffer is not None:
                obs["depth"] = state.depth_buffer[None]
            if state.automap_buffer is not None:
                obs["automap"] = state.automap_buffer
        return state, obs, done

    def _update_frame_stack(self, obs, reset: bool = False):
        if self.frame_stack.maxlen == 1:
            return obs
        else:
            if reset:
                # clear and populate frame stack
                self.frame_stack.clear()
                for _ in range(self.frame_stack.maxlen):
                    self.frame_stack.append(obs)
            else:
                self.frame_stack.append(obs)
            # return stacked observation dictionary
            obs = {
                k: np.stack([frames[k] for frames in self.frame_stack], axis=1)
                for k in obs
            }
            return obs

    def _add_bot(self):
        self.game.send_game_command("removebots")
        for i in range(self.cfg.num_bots):
            self.game.send_game_command("addbot")

    def _update_game_vars(self, reset: bool = False):
        self._game_vars_pre = deepcopy(self._game_vars)
        if self.unwrapped._state is not None:  # ensure current frame is available
            for key in self._game_var_list:
                key_name = str(key)
                self._game_vars[key_name[key_name.find(".") + 1 :]] = (
                    self.game.get_game_variable(key)
                )
            # Fix "HEALTH == -999900.0" error when respawn
            if self._game_vars["HEALTH"] < 0.0:
                self._game_vars["HEALTH"] = 0.0

    def _record_replay(self, obs, reset: bool = False):
        if reset:
            self.replay = defaultdict(list)

        if self.record:
            self.replay["tic"].append(self.game.get_episode_time())
            self.replay["frames"].append(obs["screen"])
            for k in ["labels", "depth", "automap"]:
                if k in obs:
                    self.replay[k].append(obs[k])
            self.replay["game_vars"].append(copy.deepcopy(self._game_vars))


class VizdoomMPEnv(Env):
    """
    Custom VizDoom multiplayer environment for the jku.wad DLR 25 final challenge.

    Parameters:
    - config_path (default: "doom_arena/scenarios/jku.cfg"):
    Path to the VizDoom configuration file. **Do not change** for the challenge.

    - reward_fn (default: None):
    Callable used to compute rewards. To define custom rewards, extend the `__call__`
    method in `doom_arena.reward.VizDoomReward`.

    - num_players (default: 1):
    Number of external agent players. Players connect via sockets and share the same
    game instance, with player one acting as host.

    - num_bots (default: 0):
    Number of NPC enemies in the environment.

    - bot_skill (default: 0):
    Skill level for bots, ranging from 0 to 3 (where 3 represents perfect bots).

    - discrete7 (default: True):
    Enables a simplified action space with only 7 discrete buttons.

    - episode_timeout (default: 10050):
    Maximum number of frames before the episode is terminated.

    - n_stack_frames (default: 1):
    Number of frames to include in the stacked observation buffer.

    - extra_state (default: None):
    Additional sensor observations. Can be one of the following:
        - `ObsBuffer.LABELS`: first-person view with enemies and pickups highlighted.
        - `ObsBuffer.DEPTH`: first-person depth map.
        - `ObsBuffer.AUTOMAP`: top-down view using Doom's automap.

    - doom_map (default: "ROOM"):
    Map to load in the environment. Options:
        - `TRNM`: Medium-sized map used for PvP tournaments.
        - `TRNMBIG`: Large map with multiple interconnected rooms.
        - `ROOM`: Small map for evaluation and grading; supports up to 4 bots.

    - crosshair (default: True):
    Enables the red aiming crosshair.

    - hud (default: "full"):
    Heads-up display level. Options are `"full"`, `"minimal"`, or `"off"` (or `None`).
    """

    def __init__(
        self,
        config_path: str = "doom_arena/scenarios/jku.cfg",
        reward_fn: Optional[Callable] = None,
        num_players: int = 1,
        num_bots: int = 0,
        bot_skill: int = 0,
        discrete7: bool = True,
        episode_timeout: int = 10050,
        n_stack_frames: Union[int, List[int]] = 1,
        extra_state: Optional[Union[List[ObsBuffer], List[List[ObsBuffer]]]] = None,
        doom_map: str = "ROOM",
        crosshair: Union[bool, Sequence[bool]] = True,
        hud: Optional[
            Union[str, bool, Sequence[Optional[Union[str, bool]]]]
        ] = "full",
        screen_format: Union[int, Sequence[int]] = 0,
        seed: int = 1337,
    ):
        if config_path == "doom_arena/scenarios/jku.cfg":
            assert doom_map in ["TRNM", "TRNMBIG", "ROOM"]
            if doom_map == "ROOM":
                num_bots = min(num_bots, 4)
        else:
            warn(f"Using custom untested configuration : {config_path}.")
        self.num_players = num_players
        self.num_bots = num_bots

        # extra state
        if extra_state is None or len(extra_state) == 0:
            extra_state = [None] * num_players
        else:
            if not isinstance(extra_state, (List, Tuple)):
                # wrap non-list element if single string
                extra_state = [extra_state]
            if any(isinstance(x, (List, Tuple)) for x in extra_state):
                if not all(isinstance(x, (List, Tuple)) for x in extra_state):
                    # wrap non-list elements in list
                    extra_state = [
                        [x] if not isinstance(x, (List, Tuple)) else x
                        for x in extra_state
                    ]
                assert (
                    len(extra_state) == num_players
                ), "One extra state list per player."
            else:
                extra_state = [extra_state] * num_players

        # other configs
        if not isinstance(n_stack_frames, Sequence):
            n_stack_frames = [n_stack_frames] * num_players
        if not isinstance(crosshair, Sequence):
            crosshair = [crosshair] * num_players
        if isinstance(hud, str) or not isinstance(hud, Sequence):
            hud = [hud] * num_players
        if not isinstance(screen_format, Sequence):
            screen_format = [screen_format] * num_players
        for name, values in [
            ("n_stack_frames", n_stack_frames),
            ("crosshair", crosshair),
            ("hud", hud),
            ("screen_format", screen_format),
        ]:
            if len(values) != num_players:
                raise ValueError(
                    f"{name} must contain one value per player "
                    f"({num_players} expected, got {len(values)})."
                )
        # select empty port for multiplayer
        self.port = pick_unused_port()
        # host cfg
        self.host_cfg = PlayerHostConfig(self.port)
        self.host_cfg.num_players = num_players
        # join cfg
        self.join_cfg = PlayerJoinConfig(self.port)
        # player cfg
        self.players_cfg = []
        # players
        for i in range(num_players):
            cfg = PlayerConfig()
            cfg.config_path = config_path
            cfg.player_mode = vzd.Mode.PLAYER
            cfg.screen_resolution = vzd.ScreenResolution.RES_256X192
            cfg.screen_format = vzd.ScreenFormat(screen_format[i])
            cfg.ticrate = 35
            cfg.crosshair = crosshair[i]
            cfg.respawns = True
            cfg.hud = hud[i]
            cfg.n_stack_frames = n_stack_frames[i]
            cfg.transform = transforms.Compose([to_tensor, resize, minmax, cat_dict])
            cfg.bot_skill = bot_skill
            if extra_state[i] is not None:
                if isinstance(extra_state[i][0], ObsBuffer):
                    extra_state[i] = [v.value for v in extra_state[i]]
                cfg.use_labels = ObsBuffer.LABELS.value in extra_state[i]
                cfg.use_depth = ObsBuffer.DEPTH.value in extra_state[i]
                cfg.use_automap = ObsBuffer.AUTOMAP.value in extra_state[i]
            if doom_map is not None:
                cfg.doom_map = doom_map
            cfg.episode_timeout = episode_timeout
            if i == 0:
                cfg.host_cfg = self.host_cfg
                cfg.name = "WhoAmI"
                cfg.num_bots = num_bots
            else:
                cfg.join_cfg = self.join_cfg
                cfg.name = f"P{i}"
            self.players_cfg.append(cfg)

        self.envs = []
        for i, cfg in enumerate(self.players_cfg):
            e = PlayerEnv(cfg, discrete7=discrete7, seed=seed)
            self.envs.append(e)

        if len(self.envs) == 1:
            self.env = self.envs[0]
        else:
            self.env = ParallelEnv(self.envs)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        if reward_fn is None:
            reward_fn = VizDoomReward(num_players)
        self.reward_fn = reward_fn

    def step(self, actions):
        obs, vizdoom_rwds, dones, info = self.env.step(actions)
        if len(self.envs) == 1:
            # single player
            obs = [obs]
            vizdoom_rwds = [vizdoom_rwds]
            dones = [dones]
        done = all(dones)

        rwds = []
        for player_idx in range(self.num_players):
            rwd_p = self.reward_fn(
                vizdoom_rwds[player_idx],
                self.envs[player_idx].unwrapped._game_vars,
                self.envs[player_idx].unwrapped._game_vars_pre,
                player_idx,
            )
            rwds.append(rwd_p)

        # total rewards
        rwds = [sum(rwd_p) for rwd_p in rwds]
        return obs, rwds, done, info

    def reset(self, **kwargs):
        self.reward_fn.reset()
        self.obs = self.env.reset(**kwargs)
        if len(self.envs) == 1:
            self.obs = [self.obs]
        return self.obs

    def enable_replay(self):
        print("Enabling replays!")
        for e in self.envs:
            e.record = True

    def disable_replay(self):
        print("Disabling replays!")
        for e in self.envs:
            e.record = False

    def get_player_replays(self):
        replays = {}
        for e in self.envs:
            replays[e.cfg.name] = e.replay

        return replays
