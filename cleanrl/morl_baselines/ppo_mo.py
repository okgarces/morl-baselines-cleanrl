# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from typing import Any, List

import mo_gymnasium as mo_gym
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "mo-mountaincar-v0" # "deep-sea-treasure-v0" "CartPole-v1" TODO How to fix the mountain car?
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 500
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def simplex_grid(n, step):
    """
    Generate a grid over the (n-1)-simplex.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")

    m = round(1 / step)
    if abs(m * step - 1.0) > 1e-9:
        raise ValueError("step must divide 1 exactly, e.g. 0.5, 0.25, 0.1")

    results = []

    def generate(parts_left: int, total_left: int, current: List[int]) -> None:
        if parts_left == 1:
            results.append(current + [total_left])
            return

        for value in range(total_left + 1):
            generate(parts_left - 1, total_left - value, current + [value])

    generate(n, m, [])

    return [[x * step for x in vec] for vec in results]

class UtilityFunctionLinear(nn.Module):
    def __init__(self, reward_shape=2, norm=False, weights_step=0.25, function_choice=-1, keep_scale=False):
        super().__init__()
        print('initializing linear utility function')
        self.function_choice = function_choice

        self.min_val = np.full(reward_shape, np.inf)
        self.max_val = np.full(reward_shape, -np.inf)

        self.min_val = np.full(reward_shape, -200)
        self.max_val = np.full(reward_shape, 0)

        self.params_update_flag = False
        self.norm = norm
        self.keep_scale = keep_scale
        self.reward_shape = reward_shape

        self.utility_functions = []

        weights = simplex_grid(self.reward_shape, weights_step)
        weights.append((np.ones(reward_shape) * 1 / self.reward_shape).tolist()) # add the barycentre.
        weights.append([.9, 0.00, 0.00, .1])

        for weight in weights:
            weight = np.array(weight)
            self.utility_functions.append(lambda x, w=weight: np.sum(x * np.expand_dims(w, 0), 1))

    def forward(self, xx):
        x = np.array(xx)
        if self.norm:
            self.min_val = np.minimum(x.min(0), self.min_val)
            self.max_val = np.maximum(x.max(0), self.max_val)
        if self.keep_scale:
            scale = (self.max_val - self.min_val).max()
            middle_point = (self.max_val + self.min_val) / 2
            min_input = middle_point - scale / 2
            max_input = middle_point + scale / 2
        else:
            min_input = self.min_val
            max_input = self.max_val

        inputs = np.concatenate([[min_input], [max_input], x], 0)
        if self.norm:
            inputs = (inputs - np.expand_dims(min_input, 0)) / np.expand_dims(max_input - min_input + 1e-5, 0)
        utilities = self.compute_utility(inputs)
        min_util, max_util, util = utilities[0], utilities[1], utilities[2:]
        # util = (util - min_util) / (max_util - min_util + 1e-6)
        # util *= (max_input - min_input).mean()
        return util

    def compute_utility(self, x):
        return self.utility_functions[self.function_choice](x)

class MultiEnvUtilityFunctionESR(mo_gym.wrappers.vector.MORecordEpisodeStatistics):
    """
    This class implements a multi-objective multi-environment utility function.
    This class works with MO-Gym environments.
    This class keeps the statistics of a MO-Gym environment, but returns the computed utility reward.
    Therefore, we inherit from MORecordEpisodeStatistics.
    """
    def __init__(
            self,
            venv,
            utility_function,
            discount_factor=0.99,
            reward_dim=2 # TODO ?
    ):
        super().__init__(venv)

        self.utility_function = utility_function
        self.utility_function.eval() # By default eval.

        self.total_reward = np.zeros([self.num_envs])
        self.total_discounted_reward = np.zeros([self.num_envs])

        self.reward_dim = reward_dim
        self.gamma = discount_factor

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        """Resets the environment."""
        initial_obs, info = super().reset(seed=seed, options=options)
        self.total_reward = np.zeros([self.num_envs])
        self.total_discounted_reward = np.zeros([self.num_envs])

        return initial_obs, info

    def step(self, action):
        """ Step through the environment. """
        next_obs, rewards, terminations, truncations, infos = super().step(action)

        dones = np.logical_or(terminations, truncations)

        with torch.no_grad():
            utility_reward = self.utility_function(rewards)
        self.total_reward += utility_reward
        self.total_discounted_reward += self.gamma * utility_reward

        if dones.any():
            infos['episode']['ur'] = self.total_reward.copy() # Utility Return
            infos['episode']['disc_ur'] = self.total_discounted_reward.copy()  # Utility Return

            for index_done, done in enumerate(dones):
                if done:
                    self.total_reward[index_done] = 0.0
                    self.total_discounted_reward[index_done] = 0.0

        return next_obs, utility_reward, terminations, truncations, infos

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = mo_gym.make(env_id, render_mode="rgb_array", max_episode_steps=1000, add_speed_objective=True)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = mo_gym.make(env_id, add_speed_objective=True, max_episode_steps=1000)
        # env = gym.wrappers.vector.RecordEpisodeStatistics(env)
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":

    utility_functions = {
        'linear': UtilityFunctionLinear,
    }

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.utility_function = "linear"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    reward_dim = 4
    utility_function = utility_functions[args.utility_function](reward_dim) # TODO

    envs_list = [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    envs = mo_gym.wrappers.vector.MOSyncVectorEnv(envs_list)
    envs = MultiEnvUtilityFunctionESR(envs,
                                      utility_function=utility_function,
                                      reward_dim=reward_dim)

    # assert args.num_envs == 1, "Vectorised environments not supported"
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    truncations = torch.zeros((args.num_steps, args.num_envs)).to(device)

    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    # next_done = torch.zeros(args.num_envs).to(device)

    next_terminations = torch.zeros(args.num_envs).to(device)
    next_truncations = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            # dones[step] = next_done
            terminations[step] = next_terminations
            truncations[step] = next_truncations

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_terminations, next_truncations, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_terminations, next_truncations = torch.Tensor(next_obs).to(device), torch.Tensor(next_terminations).to(device), torch.Tensor(next_truncations).to(device)


            if any(next_terminations):
                print(infos)

            # if "_final_info" in infos:
                # for info in infos["_final_info"]:
                    # if info and "episode" in info:
            if 'episode' in infos:
                episode = infos['episode']
                dones = np.logical_or(next_terminations, next_truncations)

                for index_done, done in enumerate(dones):
                    if done:
                        episode_return_env = episode['r'][index_done]
                        # This is for MORL
                        for objective, episode_return in enumerate(episode_return_env):
                            print(f"Objective={objective + 1} global_step={global_step}, episodic_return={episode_return}")
                            writer.add_scalar("charts/episodic_return", episode_return, global_step)
                            writer.add_scalar(f"charts/episodic_return/objective_{objective + 1}", episode_return, global_step)

                        episode_length_env = episode['l'][index_done]
                        writer.add_scalar(f"charts/episodic_length", episode_length_env, global_step)

                        utility_return = episode['ur'][index_done] # utility returns
                        writer.add_scalar(f"charts/utility_returns", utility_return, global_step)

                        utility_return = episode['disc_ur'][index_done]  # utility returns
                        writer.add_scalar(f"charts/discounted_utility_returns", utility_return, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_terminations
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - terminations[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
