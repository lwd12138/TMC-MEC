import numpy as np
import torch
from env_proport_reqOnly import Environment
from normalization import Normalization, RewardScaling
import math
from args import parser
import wandb
from TD3 import TD3
import utils
import os

args = parser.parse_args()

# Set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# epsilon_max = 1
# epsilon_min = 0
# eps_decay = 100000
# epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
#     -1. * frame_idx / eps_decay)

env = Environment(args)

state_dim = args.user_num
action_dim = args.user_num * 2 + args.task_num
max_action = args.max_action

slots = 400
episodes = 700
max_train_steps = slots * episodes
total_steps = 0
episode_list = []
reward_list = []
mean_reward_list = []
ca_r_list = []
co_r_list = []
la_r_list = []
avg_offRatio_list = []
hit_ratio_list = []
episode_hit_ratio = 0

agent = TD3(args, state_dim, action_dim, max_train_steps, max_action=1)
replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
state_norm = Normalization(shape=state_dim)  # state normalization

# save dir
log_root_dir = f'result/TD3/dmax/'
if not os.path.exists(log_root_dir):
    os.makedirs(log_root_dir)
log_dir = f'{args.user_num}_{args.task_num}_{args.tolerant_latency}_{args.probability_bound}'.replace(" ", "")
result_path = log_root_dir + log_dir
if not os.path.exists(result_path):
    os.makedirs(result_path)   # for saving reward data!

print("State dim: ", state_dim)
print("Action dim: ", action_dim)
print("Number of users: ", args.user_num)
print("Number of tasks: ", args.task_num)
print("Mean result size: ", np.mean(env.result_size))
print("-----Start training-----")

# wandb.init(
#     project="TD3_contrast_modify",
#     entity=None,
#     config=vars(args),
#     name="users{}_tasks{}_dmax{}_prob{}_seed{}".format(args.user_num, args.task_num, args.tolerant_latency, args.probability_bound, args.seed),
# )

for episode in range(episodes):
    if episode % 20 == 1:
        print("Worst cost:", env.worst_cost())
        print("Min S1 rate: ", np.min(env.S1_rate, axis=1))
        print("Caching policy: ", env.caching_state)
        print("Result size:", env.result_size)
        print("Hit rate: ", episode_hit_ratio)
        print("Offloading policy: ", env.off_vector)
        print("Resource allocation: ", env.server_fc_alloc)
        print("Violate probability: ", env.reliability())
    state = env.reset()
    if args.use_state_norm: # Trick 2:state normalization
        state = state_norm(state)   # Norm on this state
    reward_sum = 0
    ca_r_sum = 0
    co_r_sum = 0
    la_r_sum = 0
    off_sum = 0
    hit_ratio_sum = 0
    for slot in range(slots):
        total_steps += 1
        action = (agent.select_action(state) + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                    ).clip(-max_action, max_action)
        reward, next_state, cache_r, cost_r, latency_r = env.step(action)
        if args.use_state_norm: # Trick:state normalization
            next_state = state_norm(next_state)
        replay_buffer.add(state, action, next_state, reward)
        agent.train(replay_buffer, total_steps, args.batch_size)
        state = next_state
        reward_sum += reward
        ca_r_sum += cache_r
        co_r_sum += cost_r
        la_r_sum += latency_r
        off_sum += np.mean(env.off_vector)
        hit_ratio_sum += env.l_slot_hit_rate()

    episode_reward = reward_sum / slots
    episode_ca_r = ca_r_sum / slots
    episode_co_r = co_r_sum / slots
    episode_la_r = la_r_sum / slots
    episode_off = off_sum / slots
    episode_hit_ratio = hit_ratio_sum / slots
    print('episode =', episode, 'reward =', episode_reward, 'ca =', episode_ca_r, 'co =', episode_co_r, 'la =', episode_la_r, 'off =', episode_off)
    
    episode_list.append(episode)
    reward_list.append(episode_reward)
    ca_r_list.append(episode_ca_r)
    co_r_list.append(episode_co_r)
    la_r_list.append(episode_la_r)
    avg_offRatio_list.append(np.mean(env.off_vector))
    hit_ratio_list.append(episode_hit_ratio)

    # wandb.log({
    #     "episode": episode,
    #     "reward": episode_reward,
    #     "ca_r": episode_ca_r,
    #     "co_r": episode_co_r,
    #     "la_r": episode_la_r,
    #     "off_ratio": episode_off,
    # })

    if (episode + 1) % args.save_freq == 0:
        np.save(result_path + "/total_reward.npy", reward_list)
        np.save(result_path + "/cache_reward.npy", ca_r_list)
        np.save(result_path + "/cost_reward.npy", co_r_list)
        np.save(result_path + "/latency_reward.npy", la_r_list)
        np.save(result_path + "/average_off_ratio.npy", avg_offRatio_list)
        np.save(result_path + "/hit_ratio.npy", hit_ratio_list)

print("-----Finish training-----")
avg_50_co = np.mean(co_r_list[-50:])
avg_50_ca = np.mean(hit_ratio_list[-50:])
print("Average cost saving", avg_50_co)
print("Average hit ratio: ", avg_50_ca)