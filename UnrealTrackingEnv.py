import gym
import numpy as np
from scipy.integrate import odeint
from scipy.spatial.transform.rotation import Rotation as R
from unreal_api.environment import Environment
import traceback
import random
import cv2


def reward_tracking_vision(x, y, z, v=np.zeros((4,)), u=np.zeros((4,)), optimal_distance=2., max_dist=15., min_dist=1., exp=(1 / 3), alpha=0., beta=0., max_steps=400):
    done = False

    y_ang = np.arctan(y / x)
    z_ang = np.arctan(z / x)

    y_error = abs(y_ang / (np.pi / 4))
    z_error = abs(z_ang / (np.pi / 4))
    x_error = abs(x - optimal_distance)

    z_rew = max(0, 1 - z_error)
    y_rew = max(0, 1 - y_error)
    x_rew = max(0, 1 - x_error)

    vel_penalty = np.linalg.norm(v) / (1 + np.linalg.norm(v))
    u_penalty = np.linalg.norm(u) / (1 + np.linalg.norm(u))

    reward_track = (x_rew * y_rew * z_rew) ** exp

    reward = (reward_track - alpha * vel_penalty - beta * u_penalty) * (400 / max_steps)

    if abs(np.linalg.norm(np.array([x, y, z]))) > max_dist or abs(np.linalg.norm(np.array([x, y, z]))) < min_dist:
        done = True
        reward = -10 / (400 / max_steps)

    return reward, done


def drone_dyn(X, t, g, m, w, f):
    X = np.expand_dims(X, axis=1)
    # Variables and Parameters
    zeta = np.array([0, 0, 1]).reshape(3, 1)
    gv = np.array([0, 0, -1]).reshape(3, 1) * g
    p = X[0: 3, 0]
    v = X[3: 6, 0]
    R = X[6: 15].reshape(3, 3)

    # Drone Dynamics
    dp = v
    dv = np.dot(R, zeta) * f / m + gv
    dR = np.dot(R, sKw(w))

    #
    dX = np.concatenate((dp.reshape(-1, 1), dv, dR.reshape(-1, 1)), axis=0).squeeze()
    return dX


def sKw(x):
    Y = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype=np.float64)
    return Y


class UnrealTrackingEnv(gym.Env):
    def __init__(self,
                 rank=0,
                 test=False,
                 start_tracker_port=9734,
                 start_target_port=9735,
                 render=False,
                 dr=True,
                 ts=0.05,
                 observation_buffer_length=3):
        super(UnrealTrackingEnv, self).__init__()
        print('UnrealTrackingEnv rank: ', rank)
        self.rank = rank
        self.test = test
        self.dr = dr
        self.observation_buffer_length = observation_buffer_length
        self.action_limit = 4
        self.action_space = gym.spaces.Box(-self.action_limit, self.action_limit, shape=(4,), dtype=np.float32)

        self.optimal_distance = 0.5
        self.image_shape = (3, 224, 224)

        critic_obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        acotr_obs_space = gym.spaces.Box(0.0, 1.0, shape=(self.observation_buffer_length,) + self.image_shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'actor': acotr_obs_space,
            'critic': critic_obs_space
        })

        self.state = None
        # Actor History
        self.actor_state = None

        self.reward = 0
        self.done = 0
        self.episode_steps = 0
        self.info = {}

        # SIMULATION PARAMETERS
        self.g = 9.8  # Gravitational Acceleration[m / s ^ 2]
        self.Tin = 0  # Initial time[s]
        self.Ts = ts  # Sampling time[s] 0.05
        self.T_target = 0.0
        self.Ts_target = ts

        # EPISODE
        self.episode_time = 40  # s
        self.max_episode_steps = int(self.episode_time / self.Ts)
        self.stop_step = random.randint(0, self.max_episode_steps)
        self.stop_duration = 10 / self.Ts

        # DRONE PARAMETERS
        self.m = 1  # # Tracker Mass[kg]
        self.Xin = None
        self.fk = self.m * self.g

        # TARGET MOVEMENT
        self.move_target = True
        self.pr0 = np.array([0.0, 0.0, 0.0])
        self.prout = np.array([0.0, 0.0, 0.0])
        self.vrout = np.array([0.0, 0.0, 0.0])
        self.arout = np.array([0.0, 0.0, 0.0])
        self.Rrout = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
        self.a1, self.a2, self.a3 = None, None, None
        self.phi1, self.phi2, self.phi3 = None, None, None
        self.ws1, self.ws2, self.ws3 = None, None, None
        self.reset_target()

        # Unreal
        self.connected = False
        self.envs = []
        self.port_tracker = start_tracker_port + (self.rank * 2)
        self.port_target = start_target_port + (self.rank * 2)
        self.sensor_settings = {
            "RGBCamera": {
                "width": self.image_shape[1],
                "height": self.image_shape[2],
                "channels": "RGB",
                "FOV": 90,
                "show": self.test
            },
            "GPS": {}
        }
        self.action_manager_settings = {  # X Y Z PYR
            "CoordinateActionManager": {
                "command_dict": {
                    "MOVETO": 0
                },
                "settings": {
                }
            }
        }
        self.reset_manager_settings = {
            "EnvResetManager": {}
        }
        self.observation_list_target = []
        self.observation_list_tracker = ["RGBCamera"]

        self.ue_to_meters = 100
        self.offset = [0, 100 * self.rank + 800 * self.rank, 100]

        try:
            if len(self.envs) == 0:
                self.envs.append(
                    Environment(self.port_tracker, self.sensor_settings, self.action_manager_settings, reset_manager_settings=self.reset_manager_settings, render=render))
                self.envs.append(
                    Environment(self.port_target, self.sensor_settings, self.action_manager_settings, render=render))
                self.connected = True
        except:
            print("ERROR: Cannot connect to Unreal")
            traceback.print_exc()

    def reset_target(self, p=None, rand=False):
        self.T_target = 0
        if self.move_target:
            self.a1 = 1 + random.random() * 0.5  # 30
            self.a2 = 1 + random.random() * 0.5  # 30
            self.a3 = 1 + random.random() * 1.5  # 3

            k1 = 6 + random.random() * 6
            k2 = 6 + random.random() * 6
            k3 = 6 + random.random() * 6

            f1 = 1 / (k1 * abs(self.a1))
            f2 = 1 / (k2 * abs(self.a2))
            f3 = 1 / (k3 * abs(self.a3))

            self.phi1 = -np.pi / 2 + random.random() * np.pi / 2
            self.phi2 = -np.pi / 2 + random.random() * np.pi / 2
            self.phi3 = -np.pi / 2 + random.random() * np.pi / 2

            self.ws1 = 2 * np.pi * f1
            self.ws2 = 2 * np.pi * f2
            self.ws3 = 2 * np.pi * f3

            if p is not None:
                if rand:
                    fov = np.pi / 4
                    xdis = random.random() + self.optimal_distance + 0.1
                    yang = random.random() * fov - fov / 2
                    zang = random.random() * fov - fov / 2
                    ydis = xdis * np.tan(yang)
                    zdis = xdis * np.tan(zang)

                    self.pr0 = np.array([p[0] + xdis, p[1] + ydis, p[2] + zdis])
                else:
                    self.pr0 = np.array([p[0] + self.optimal_distance, p[1], p[2]])
            else:
                self.pr0 = np.array([0.0, 0.0, 0.0])

        self.prout = self.pr0
        self.vrout = np.array([0.0, 0.0, 0.0])
        self.arout = np.array([0.0, 0.0, 0.0])
        self.Rrout = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

    def step(self, u):
        # Define a control input for the tracker
        u = u.reshape(self.action_space.shape[0], 1)

        self.episode_steps += 1
        if self.move_target and (self.episode_steps <= self.stop_step or self.episode_steps > (self.stop_step + self.stop_duration)):
            self.prout = np.array([self.a1 * np.sin(self.phi1 + self.ws1 * self.T_target) - self.a1 * np.sin(self.phi1) + self.pr0[0],
                                   self.a2 * np.sin(self.phi2 + self.ws2 * self.T_target) - self.a2 * np.sin(self.phi2) + self.pr0[1],
                                   self.a3 * np.sin(self.phi3 + self.ws3 * self.T_target) - self.a3 * np.sin(self.phi3) + self.pr0[2]])
            self.vrout = np.array([self.a1 * self.ws1 * np.cos(self.phi1 + self.T_target * self.ws1),
                                   self.a2 * self.ws2 * np.cos(self.phi2 + self.T_target * self.ws2),
                                   self.a3 * self.ws3 * np.cos(self.phi3 + self.T_target * self.ws3)])
            self.arout = np.array([-self.a1 * self.ws1 ** 2 * np.sin(self.phi1 + self.T_target * self.ws1),
                                   -self.a2 * self.ws2 ** 2 * np.sin(self.phi2 + self.T_target * self.ws2),
                                   -self.a3 * self.ws3 ** 2 * np.sin(self.phi3 + self.T_target * self.ws3)])

            self.T_target += self.Ts_target
        else:
            self.vrout = np.array([0, 0, 0])
            self.arout = np.array([0, 0, 0])

        # Integrate dynamics
        t = [self.Tin, self.Tin + self.Ts]
        # TRACKER #
        u = u.reshape(self.action_space.shape[0], )

        w = u[:3]
        f = (u[3] * 5 + 20.2) / 2
        self.fk = f

        Xout = odeint(drone_dyn, self.Xin.squeeze(), t, args=(self.g, self.m, w, self.fk))  # X, t, g, m, w, f

        Xout = Xout[-1, :].T
        Tout = t[-1]

        # Tracker output variables
        pout = Xout[0: 3]
        vout = Xout[3: 6]
        Rout = Xout[6: 15].reshape(3, 3)

        zeta = np.array([0, 0, 1]).reshape(3, 1)
        gv = np.array([0, 0, -1]).reshape(3, 1) * self.g
        aout = (np.dot(Rout, zeta) * (self.fk / self.m) + gv).reshape(3, )

        # X Y Z of the target wrt to the tracker at time Tout
        x, y, z = np.dot(np.dot(np.array([1, 0, 0]), Rout.T), self.prout - pout), \
                  np.dot(np.dot(np.array([0, 1, 0]), Rout.T), self.prout - pout), \
                  np.dot(np.dot(np.array([0, 0, 1]), Rout.T), self.prout - pout)

        v_x, v_y, v_z = np.dot(np.dot(np.array([1, 0, 0]), Rout.T), self.vrout - vout), \
                        np.dot(np.dot(np.array([0, 1, 0]), Rout.T), self.vrout - vout), \
                        np.dot(np.dot(np.array([0, 0, 1]), Rout.T), self.vrout - vout)

        a_x, a_y, a_z = np.dot(np.dot(np.array([1, 0, 0]), Rout.T), self.arout - aout), \
                        np.dot(np.dot(np.array([0, 1, 0]), Rout.T), self.arout - aout), \
                        np.dot(np.dot(np.array([0, 0, 1]), Rout.T), self.arout - aout)

        # Unreal Rendering
        # Target
        p_target = self.prout.reshape(3,)
        R_target = self.Rrout
        qx, qy, qz, qw = R.from_matrix(R_target).as_quat()
        action_target = [p_target[0] * self.ue_to_meters + self.offset[0], p_target[1] * self.ue_to_meters + self.offset[1], p_target[2] * self.ue_to_meters + self.offset[2], qx, qy, qz, qw]
        self.envs[1].env_step(action_target, self.observation_list_target)

        # Tracker
        p_tracker = self.Xin[0: 3].reshape(3,)
        R_tracker = self.Xin[6: 15].reshape(3, 3)
        qx, qy, qz, qw = R.from_matrix(R_tracker).as_quat()
        action_tracker = [p_tracker[0] * self.ue_to_meters + self.offset[0], p_tracker[1] * self.ue_to_meters + self.offset[1], p_tracker[2] * self.ue_to_meters + self.offset[2], qx, qy, qz, qw]

        _, obs = self.envs[0].env_step(action_tracker, self.observation_list_tracker)

        image = cv2.cvtColor((obs[0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        norm_image = (image.astype(np.float32) / 255).reshape(self.image_shape)

        state_critic = np.array([x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]).reshape((1, 9))
        state_actor = np.expand_dims(norm_image, axis=0)
        if self.actor_state is None:
            self.actor_state = [state_actor] * self.observation_buffer_length
        self.actor_state.append(state_actor)
        self.actor_state.pop(0)
        current_state_actor = np.concatenate(self.actor_state, axis=0)
        self.state = {
            'actor': current_state_actor,
            'critic': state_critic
        }

        # SubProcVecEnv Fix
        self.state['critic'] = self.state['critic'].flatten()


        # Reward Computation
        self.reward, self.done = reward_tracking_vision(x, y, z,
                                                        v=np.array([v_x, v_y, v_z]),
                                                        u=(u - np.array([0, 0, 0, self.m * self.g]) / np.array([self.action_limit, self.action_limit, self.action_limit, self.action_limit * 5 / 2])),
                                                        optimal_distance=self.optimal_distance,
                                                        min_dist=0.3,
                                                        max_dist=3.0,
                                                        alpha=0.4,
                                                        beta=0.4,
                                                        max_steps=self.max_episode_steps)

        # Update loop states
        self.Xin = Xout
        self.Tin = Tout

        if self.episode_steps >= self.max_episode_steps:
            self.done = True

        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.episode_steps = 0
        self.stop_step = random.randint(0, self.max_episode_steps)
        self.actor_state = None

        # INITIAL CONDITIONS
        # Tracker
        pin = np.zeros((3, 1))
        vin = np.array([0, 0, 0]).reshape(3, 1)  # Tracker Initial Velocity[m]
        Rin = np.eye(3)  # Tracker Initial Attitude(Body->Inertial)
        self.fk = self.g * self.m
        self.Xin = np.concatenate((pin, vin, Rin.reshape(-1, 1)), axis=0)

        self.Tin = 0

        # Target
        self.reset_target(pin.reshape((3,)), rand=False)

        # Domain Randomization
        if self.dr:
            self.envs[0].reset([], {'scale': 0.25})

        self.step(np.array([0.0, 0.0, 0.0, self.fk]))
        self.done = False

        return self.state

    def render(self, mode='human'):
        pass
