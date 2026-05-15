import enum
import vizdoom as vzd
import os


class PlayerHostConfig(object):
    def __init__(self, port, num_players=2):
        self.num_players = num_players
        self.port = port
        print("Host {}".format(self.port))


class PlayerJoinConfig(object):
    def __init__(self, port):
        self.join_ip = "127.0.0.1"
        self.port = port
        print("Player {}".format(self.port))


class PlayerConfig(object):
    def __init__(self):
        self.config_path = None
        self.player_mode = None
        self.screen_resolution = None
        self.screen_format = None
        self.ticrate = None
        self.episode_timeout = None
        # custom character
        self.name = None
        self.colorset = None
        self.crosshair = None
        self.hud = "full"
        # observation settings
        self.transform = None
        self.use_depth = False
        self.use_labels = False
        self.use_automap = False
        # game variables
        self.available_game_variables = [
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.AMMO3,
            # additional
            vzd.GameVariable.FRAGCOUNT,
            vzd.GameVariable.ARMOR,
            vzd.GameVariable.HITCOUNT,
            vzd.GameVariable.HITS_TAKEN,
            vzd.GameVariable.DEAD,
            vzd.GameVariable.DEATHCOUNT,
            vzd.GameVariable.DAMAGECOUNT,
            vzd.GameVariable.DAMAGE_TAKEN,
            vzd.GameVariable.KILLCOUNT,
            vzd.GameVariable.SELECTED_WEAPON,
            vzd.GameVariable.SELECTED_WEAPON_AMMO,
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.ANGLE, #added
        ]
        # game buttons
        # NOTE: order is impotant
        self.available_buttons = [
            vzd.Button.MOVE_FORWARD,
            vzd.Button.ATTACK,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.JUMP,
            # vzd.Button.USE,
            # vzd.Button.MOVE_BACKWARD,
        ]

        # game map
        self.doom_map = None
        self.doom_skill = 1
        self.bot_skill = 0

        self.n_stack_frames = 1
        self.num_bots = 0
        self.respawns = False

        self.host_cfg = None
        self.join_cfg = None


def player_host_setup(game, host_config):
    game.add_game_args(
        " ".join(
            [
                "-host {}".format(host_config.num_players),
                "-port {}".format(host_config.port),
                "-netmode 0",
                "-deathmatch",
                "+sv_spawnfarthest 1",
                "+viz_nocheat 0",
            ]
        )
    )
    return game


def player_join_setup(game, join_config):
    game.add_game_args(
        " ".join(
            [
                "-join {}".format(join_config.join_ip),
                "-port {}".format(join_config.port),
            ]
        )
    )
    return game


def player_setup(game, player_config: PlayerConfig):
    if player_config.config_path is not None:
        game.load_config(player_config.config_path)

    if player_config.player_mode is not None:
        game.set_mode(player_config.player_mode)
    if player_config.screen_resolution is not None:
        game.set_screen_resolution(player_config.screen_resolution)
    if player_config.screen_format is not None:
        game.set_screen_format(player_config.screen_format)
    if player_config.ticrate is not None:
        game.set_ticrate(player_config.ticrate)
    if player_config.episode_timeout is not None:
        game.set_episode_timeout(player_config.episode_timeout)
    if player_config.episode_timeout is not None:
        game.set_episode_timeout(player_config.episode_timeout)

    if player_config.doom_map is not None:
        game.set_doom_map(player_config.doom_map)

    if player_config.doom_skill is not None:
        game.set_doom_skill(player_config.doom_skill)

    # custom game variables
    if player_config.available_game_variables is not None:
        game.set_available_game_variables(player_config.available_game_variables)

    # custom game buttons
    if player_config.available_buttons is not None:
        game.set_available_buttons(player_config.available_buttons)

    # hud
    if player_config.hud != "full":
        if player_config.hud == "minimal":
            game.set_render_minimal_hud(True)
        if player_config.hud in [False, None, "none"]:
            game.set_render_hud(False)

    game.set_render_decals(False)
    game.set_render_messages(False)

    # depth
    if player_config.use_depth:
        game.set_depth_buffer_enabled(True)
    # segmented
    if player_config.use_labels:
        game.set_labels_buffer_enabled(True)
    # automap
    if player_config.use_automap:
        game.set_automap_buffer_enabled(True)
        game.set_automap_mode(vzd.AutomapMode.OBJECTS)
        game.set_automap_rotate(True)
        game.set_automap_render_textures(False)

    game.set_console_enabled(False)

    if player_config.name is not None:
        game.add_game_args("+name {}".format(player_config.name))
    if player_config.colorset is not None:
        game.add_game_args("+colorset {}".format(player_config.colorset))

    if player_config.crosshair is not None:
        game.set_render_crosshair(player_config.crosshair)

    if player_config.respawns:
        # respawn automatically after death
        game.add_game_args("+sv_forcerespawn 1")
        # invulnerable for two second after spawning
        game.add_game_args("+sv_respawnprotect 1")
        # seconds between respawns
        game.add_game_args(f"+viz_respawn_delay 5")
    return game


def mp_game_setup(game, bot_skill: int = 0):
    # respawn items
    game.add_game_args("+altdeath 1")
    # no crouching
    game.add_game_args("+sv_nocrouch 1")
    # no monsters
    game.add_game_args("+sv_nomonsters 1")
    game.add_game_args("+nomonsters 1")
    if bot_skill == 0:
        # easy bots
        game.add_game_args(f"+viz_bots_path {os.getcwd()}/doom_arena/bots/easy.cfg")
    if bot_skill == 1:
        pass
    if bot_skill >= 2:
        # hard bots
        game.add_game_args(f"+viz_bots_path {os.getcwd()}/doom_arena/bots/hard.cfg")
    return game


class ObsBuffer(enum.Enum):
    LABELS = "labels"
    DEPTH = "depth"
    AUTOMAP = "automap"
