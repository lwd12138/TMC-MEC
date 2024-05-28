import numpy as np
import torch
from env_proport_reqOnly import Environment
from args import parser
import wandb
from SAC import SAC
import utils
import os

args = parser.parse_args()

# Set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

env = Environment(args)

state_dim = args.user_num
action_dim = args.user_num * 2 + args.task_num
max_action = args.max_action

slots = 400
episodes = 700
max_train_steps = slots * episodes
episode_list = []
reward_list = []
ca_r_list = [] 
co_r_list = []
la_r_list = []
avg_offRatio_list = []
worst_cost_list = []
hit_ratio_list = []
episode_hit_ratio = 0

agent = SAC(state_dim, action_dim, max_action=1)
replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

# save dir
log_root_dir = f'result/SAC/dmax/'
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
#     project="SAC_random_cache_modify",
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
    reward_sum = 0
    ca_r_sum = 0
    co_r_sum = 0
    la_r_sum = 0
    off_sum = 0
    hit_ratio_sum = 0
    for slot in range(slots):
        action = agent.choose_action(state)
        reward, next_state, cache_r, cost_r, latency_r = env.step(action)
        replay_buffer.add(state, action, next_state, reward)
        agent.learn(replay_buffer)
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
    avg_offRatio_list.append(episode_off)
    worst_cost_list.append(env.worst_cost())
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
        np.save(result_path + "/worst.npy", worst_cost_list)
        np.save(result_path + "/hit_ratio.npy", hit_ratio_list)

print("-----Finish training-----")
avg_50_co = np.mean(co_r_list[-50:])
print("Average cost saving", avg_50_co)