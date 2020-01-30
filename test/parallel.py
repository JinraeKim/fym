import numpy as np
import time
import multiprocessing as mp
import ray

import fym
import fym.core as core
import fym.models.aircraft as aircraft
from fym.agents.LQR import clqr


class Lqr:
    Q = np.diag([50, 100, 100, 50, 0, 0, 1])
    R = np.diag([0.1, 0.1])

    def __init__(self, sys):
        self.K = clqr(sys.A, sys.B, self.Q, self.R)[0]

    def select_action(self, x):
        return np.hstack([-self.K.dot(x), ])


class Aircraft(core.BaseEnv):
    def __init__(self, logging_off=False, rand_init=True):
        super().__init__(
            systems_dict={
                "f16": aircraft.F16LinearLateral([1, 0, 0, 0, 0, 0, 0]),
            },
            dt=0.01,
            max_t=10,
            logging_off=logging_off,
        )
        self.rand_init = rand_init
        self.reward_weight = 1
        # self.controller = Lqr(self.systems_dict["f16"])
        # self.controller_expert = Lqr(self.systems_dict["f16"])

    def reset(self):
        super().reset()
        f16, = self.systems
        x = f16.state
        if self.rand_init:
            x = x + (
                np.array([1, 20, 20, 6, 80, 80, 0])
                * np.random.uniform(-1.0, 1.0)
            )

        return x

    def observation(self):
        return self.observe_flat()

    def get_reward(self, x):
        k = self.reward_weight
        reward = - k * np.linalg.norm(x, 2)
        return reward

    def step(self, action):
        done = self.clock.time_over()
        self.update(action)
        f16, = self.systems
        x = f16.state
        reward = self.get_reward(x)
        info = {
            'time': self.clock.get(),
            'state': x,
            'action': action,
            'reward': reward,
        }
        return self.observation(), reward, done, info

    def get_control(self, x):
        u = self.controller.select_action(x)
        return u

    def set_dot(self, t, action):
        f16, = self.systems
        x = f16.state
        # u = self.get_control(x)
        f16.dot = f16.deriv(f16.state, action)


def collect_samples(obs, env, agent):
    obs_list = []
    action_list = []
    masks = []
    reward_list = []

    # obs = env.reset()
    # obs = np.zeros(7)
    while True:
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)

        obs_list.append(obs)
        action_list.append(action)
        masks.append(0 if done else 1)
        reward_list.append(reward)

        obs = next_obs

        if done:
            break

    return np.stack(obs_list), np.stack(action_list), masks, reward_list


@ray.remote
def ray_collect_samples(obs, env, agent):
    obs_list = []
    action_list = []
    masks = []
    reward_list = []

    # obs = env.reset()
    # obs = np.zeros(7)
    while True:
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)

        obs_list.append(obs)
        action_list.append(action)
        masks.append(0 if done else 1)
        reward_list.append(reward)

        obs = next_obs

        if done:
            break

    return np.stack(obs_list), np.stack(action_list), masks, reward_list


def parallel_collect_samples(env, agent, num=2):
    pool = mp.Pool()
    args_list = [(env, agent) for _ in range(num)]
    for i in range(num):
        args_list[i][0].reset()
    # results = pool.starmap_async(collect_samples, args_list)
    results = [pool.apply(collect_samples, args=(args_list[i][0], args_list[i][1])) for _ in range(num)]
    return results


if __name__ == "__main__":
    seed = 1
    # Seeding
    np.random.seed(seed)
    num_iter = 2

    # not parallel
    env = Aircraft(logging_off=True, rand_init=False)
    agent = Lqr(env.systems_dict['f16'])
    obs = env.reset()
    results_original = collect_samples(obs, env, agent)

    # parallel - ray
    obs = env.reset()

    ray.init()
    for _ in range(num_iter):
        inputs = ray.put(env, agent)
        futures = [ray_collect_samples.remote(obs, env, agent) for _ in range(1)]
        results_ray = ray.get(futures)

    print("original = {}".format(results_original[0]))
    print("ray = {}".format(results_ray[0][0]))
    # # parallel - multiprocessing
    # env = Aircraft(logging_off=True)
    # agent = Lqr(env.systems_dict['f16'])
    # parallel_collect_samples(env, agent, num=num_iter)
