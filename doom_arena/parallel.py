"""
copied from PySC2 code
"""

import functools
from concurrent import futures

import gymnasium as gym
from gymnasium.spaces import Tuple


class RunParallel(object):
    """Run all funcs in parallel."""

    def __init__(self, timeout=None):
        self._timeout = timeout
        self._executor = None
        self._workers = 0

    def run(self, funcs):
        """Run a set of functions in parallel, returning their results.

        Make sure any function you pass exits with a reasonable timeout. If it
        doesn't return within the timeout or the result is ignored due an exception
        in a separate thread it will continue to stick around until it finishes,
        including blocking process exit.

        Args:
          funcs: An iterable of functions or iterable of args to functools.partial.

        Returns:
          A list of return values with the values matching the order in funcs.

        Raises:
          Propagates the first exception encountered in one of the functions.
        """
        funcs = [f if callable(f) else functools.partial(*f) for f in funcs]
        if len(funcs) == 1:  # Ignore threads if it's not needed.
            return [funcs[0]()]
        if len(funcs) > self._workers:  # Lazy init and grow as needed.
            self.shutdown()
            self._workers = len(funcs)
            self._executor = futures.ThreadPoolExecutor(self._workers)
        futs = [self._executor.submit(f) for f in funcs]
        done, not_done = futures.wait(futs, self._timeout, futures.FIRST_EXCEPTION)
        # Make sure to propagate any exceptions.
        for f in done:
            if not f.cancelled() and f.exception() is not None:
                if not_done:
                    # If there are some calls that haven't finished, cancel and recreate
                    # the thread pool. Otherwise we may have a thread running forever
                    # blocking parallel calls.
                    for nd in not_done:
                        nd.cancel()
                    self.shutdown(False)  # Don't wait, they may be deadlocked.
                raise f.exception()
        # Either done or timed out, so don't wait again.
        return [f.result(timeout=0) for f in futs]

    def shutdown(self, wait=True):
        if self._executor:
            self._executor.shutdown(wait)
            self._executor = None
            self._workers = 0

    def __del__(self):
        self.shutdown()


class ParallelEnv(gym.Env):
    """Vectorize a list of environments as a single environment."""

    def __init__(self, envs):
        self._spectator_envs = [e for e in envs if e.is_spectator]
        self._player_envs = [e for e in envs if not e.is_spectator]

        self.observation_space = Tuple([e.observation_space for e in self._player_envs])
        self.action_space = Tuple([e.action_space for e in self._player_envs])

        self._par = RunParallel()

    def reset(self):
        observations = self._par.run((e.reset) for e in self.envs)
        return observations

    def step(self, actions):
        ret = self._par.run((e.step, act) for e, act in zip(self._player_envs, actions))
        observations, rewards, dones, infos = [item for item in zip(*ret)]
        return observations, rewards, dones, {}

    def close(self):
        self._par.run((e.close) for e in self._player_envs)

    @property
    def envs(self):
        return self._spectator_envs + self._player_envs
