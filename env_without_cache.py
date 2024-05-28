import numpy as np

class Environment:
    '''
    This class implements the wireless computing environment
    '''
    def __init__(self, args):
        self.user_num = args.user_num                       # 用户数量
        self.user_fc = args.user_frequency                  # 用户设备计算能力（CPU频率）
        self.user_kappa = args.user_e_coe                   # 用户设备的能耗系数
        self.user_power = args.user_power                   # 用户的传输功率
        self.server_fc = args.server_frequency              # 服务器计算能力（CPU频率，30GHz）
        self.server_kappa = args.server_e_coe               # 服务器的能耗系数
        self.server_c = args.server_cache_size              # 服务器缓存容量

        self.task_num = args.task_num                       # 任务数量
        self.task_comp_load = args.task_computation_load    # 任务每兆比特计算量（100,000,000 cycles/Mbit = 100 cycles/bit）
        self.task_max = args.task_max                       # 任务大小的最大值
        self.task_min = args.task_min                       # 任务大小的最小值
        # self.task_size = args.task_size                     # 任务大小
        self.result_max = args.task_result_max              # 任务计算结果大小最大值
        self.result_min = args.task_result_min              # 任务计算结果大小最小值

        self.s_slot_tau = args.s_slot_tau                   # 时隙长度
        self.s_slot_num = args.s_slot_num                   # 每个大时隙（step）中的小时隙个数

        self.bandwidth = args.bandwidth                     # 通信带宽
        self.sigma2 = args.sigma2                           # 噪声功率
        self.BS_pos = args.BS_pos                           # 基站坐标(500, 500)

        self.weight_user_ec = args.weight_user_ec           # 用户的能耗成本权重
        self.weight_BS_ec = args.weight_BS_ec               # 基站的能耗成本权重

        self.action_dim = self.user_num + self.task_num     # 动作空间维度
        self.max_action = args.max_action                   # 动作能取到的最大值（本场景所有动作都只能取0或1）

        self.probability_bound = args.probability_bound     # 时延违反概率阈值
        self.tolerant_latency = args.tolerant_latency       # 最大可容忍时延dmax

        # 路径损耗rho和系数alpha；信道质量越差则alpha越大；所有用户都相同
        self.pathloss_rho = 0.001                           # 对应-30dB，一般可对应-20dB或-30dB
        self.pathloss_alpha = 3.5                           # 一般为3.5或4

        # self.zipf_alpha = args.zipf_alpha                   # Zipf分布参数α
        # self.zipf_R = args.zipf_R                           # Zipf分布参数，表示不请求的概率
        self.zipf_N = args.zipf_N                                           # Zipf分布参数，表示请求可能转移到index相邻的N个任务
        self.zipf_alpha = np.random.choice([0.7, 0.8], size=self.user_num)  # Zipf分布参数α，只能取0.7或0.8
        self.zipf_R = np.random.choice([0.1, 0.2], size=self.user_num)      # Zipf分布参数，表示不请求的概率，只能取0.1或0.2

        self.result_size = np.random.randint(self.result_min, self.result_max + 1, size=self.task_num)
        # self.result_size = np.random.uniform(self.result_min, self.result_max, size=self.task_num)

        # 用户在一个大时隙内的请求矩阵。每行第一个元素存储上一个大时隙的最后一个小时隙的请求，用于后续计算
        self.user_req = np.random.randint(0, self.task_num + 1, size=(self.user_num, self.s_slot_num + 1))
        # self.temp_req = np.copy(self.user_req)

        # 初始化用户到基站的距离
        self.BS_position = np.array([self.BS_pos, self.BS_pos])
        self.users_position = np.array(
                    [[np.random.randint(self.BS_pos - 100, self.BS_pos + 101), np.random.randint(self.BS_pos - 100, self.BS_pos + 101)] 
                    for _ in range(self.user_num)])
        self.distances = np.linalg.norm(self.users_position - self.BS_position, axis=1).reshape((-1, 1))    # 转化为列向量，方便信道增益的计算

        # 初始化信道增益矩阵
        self.channel_gain_sample()
        
    
    # 重置模拟环境中的各个参数
    def reset(self):
        self.state = self.user_req[:, -1]
        return self.state
        
    def user_request_transfer(self):
        '''
        用户请求转移
        可得到一个大时隙内所有用户的请求矩阵
        '''
        # 计算zipf分布的分母
        # zipf_sum = 0
        zipf_sum = np.zeros(self.user_num)
        for i in range(self.task_num):
            for user_index in range(self.user_num):
                zipf_sum[user_index] = zipf_sum[user_index] + 1/((i + 1)**self.zipf_alpha[user_index])
        for user_index in range(self.user_num):
            slot_pre = 0
            self.user_req[user_index][0] = self.user_req[user_index][-1]  # 存储上一个大时隙最后一个小时隙的请求
            for slot_index in range(1, self.s_slot_num+1):
                req_prob = np.zeros(self.task_num+1, dtype=float)
                req_prob[0] = self.zipf_R[user_index]  # request nothing
                if self.user_req[user_index][slot_pre] == 0:
                    for req_task_index in range(self.task_num):
                        req_prob[req_task_index + 1] = (1-self.zipf_R[user_index]) * \
                        (1/((req_task_index + 1)**self.zipf_alpha[user_index])) / zipf_sum[user_index]
                else:
                    for ind in range(self.zipf_N):
                        index_com = ind - int(0.5*self.zipf_N) + self.user_req[user_index][slot_pre]
                        if index_com <= self.task_num and index_com > 0:
                            index = int(index_com)
                        elif index_com > self.task_num:
                            index = int(index_com - self.task_num)
                        elif index_com <= 0:
                            index = int(index_com + self.task_num)
                        req_prob[index] = (1 - self.zipf_R[user_index]) * (1 / self.zipf_N)    
                self.user_req[user_index][slot_index] = np.random.choice(range(0, self.task_num+1), p=req_prob)
                slot_pre = slot_index

    def channel_gain_sample(self):
        '''
        对每个小时隙的信道增益进行采样
        '''
        gain = np.random.rayleigh(scale=np.sqrt(0.5), size=(self.user_num, self.s_slot_num))
        self.channel_gain = gain * np.sqrt(self.pathloss_rho * np.power(self.distances, -self.pathloss_alpha))     # 信道增益矩阵

    def step(self, action):
        '''
        奖励：缓存奖励 + 能耗节约奖励 + 时延违反惩罚 
        下一个状态：用户在下一个大时隙的最后一个小时隙的请求
        '''
        action = self.preprocess(action)                                    # 将缓存的动作转化为0，1离散值，将卸载的动作映射到0-1之间
        self.caching_state = action[:self.task_num]                         # 动作向量前面为缓存策略，维度为任务数量，注意下标是从0开始的
        self.off_vector = action[self.task_num:self.user_num+self.task_num] # 动作向量中间为卸载策略，维度为用户数量
        self.server_fc_alloc = action[-self.user_num:]                      # 动作向量后面为计算资源分配策略，维度为用户数量
        self.user_request_transfer()                                        # 用户请求矩阵转移
        # self.channel_gain_sample()                                        # 用户信道增益采样，关掉表示用户的信道增益保持不变

        # 每个大时隙都对任务大小进行采样
        self.task_size = np.random.randint(self.task_min, self.task_max + 1, size=self.task_num)
        
        # 第一跳速率，单位为Mbit/s
        self.S1_rate = self.bandwidth * np.log2(1 + self.user_power * np.power(self.channel_gain, 2) / self.sigma2)
        # 分别计算每个用户在一个大时隙内的期望速率
        self.S1_rate_expectation = np.mean(self.S1_rate, axis=1)
        # 第二跳速率在大时隙内为常速率
        self.S2_rate = self.server_fc_alloc * 1e9 / self.task_comp_load

        cache_reward = self.l_slot_hit_rate() + self.cache_limit()                      # 缓存奖励
        fc_reward = self.fc_limit()                                                     # 计算资源溢出惩罚
        cost_reward = self.worst_cost() - self.total_cost()                             # 能耗节约奖励
        latency_reward = self.latency()                                                 # 时延惩罚
        reward = cache_reward + fc_reward + cost_reward + latency_reward                # 总奖励
        # reward = cache_reward + cost_reward + latency_reward

        self.state = self.user_req[:, -1]
        return reward, self.state, cache_reward, cost_reward, latency_reward
    
    # 将缓存的动作转化为0，1离散值，将卸载的动作映射到0-1之间
    def preprocess(self, action):
        action = np.clip(action, -self.max_action, self.max_action)
        # 缓存策略
        action[:self.task_num] = 0
        # 卸载策略
        action[self.task_num : self.task_num+self.user_num] = action[self.task_num : self.task_num+self.user_num] * 0.5 + 0.5
        # 计算资源分配策略
        temp = self.server_fc / self.user_num / 2 * 1e-9
        action[-self.user_num:] = action[-self.user_num:] * temp + temp
        # action[-self.user_num:] = 3

        return action
    
    # 缓存命中率奖励
    def l_slot_hit_rate(self):
        hit = 0
        not_hit = 0
        for user_index in range(self.user_num):
            for slot_index in range(1, self.s_slot_num+1):
                if self.user_req[user_index][slot_index] != 0:
                    if self.caching_state[self.user_req[user_index][slot_index]-1] == 1:
                        hit += 1
                    else:
                        not_hit += 1
        hit_rate = hit / (hit + not_hit)
        hit_rate_reward = 100 * hit_rate
        return hit_rate_reward

    # 缓存溢出惩罚
    def cache_limit(self):
        cache_usage = np.sum(np.multiply(self.caching_state, self.result_size))
        cache_punish = 0
        if cache_usage > self.server_c:
            cache_punish = -50 * (cache_usage - self.server_c)
        return cache_punish

    # 计算资源溢出惩罚
    def fc_limit(self):
        fc_punish = 10 * np.sum(self.server_fc_alloc)
        fc_punish = 0
        fc_usage = np.sum(self.server_fc_alloc)
        if fc_usage > self.server_fc:
            fc_punish -= 50 * (fc_usage - self.server_fc)
        return fc_punish
    
    # 总成本
    def total_cost(self):
        '''
        总成本：能耗成本 + 缓存成本
        能耗包括用户端能耗和基站端能耗，用户端能耗包括本地计算能耗和传输能耗，基站端能耗为边缘服务器计算能耗
        '''
        user_energy_cost = 0
        BS_energy_cost = 0
        for user_index in range(self.user_num):
            for slot_index in range(1, self.s_slot_num+1):
                # 只有用户在该小时隙请求了任务并且该任务没有被缓存时，才产生能耗成本
                if self.user_req[user_index][slot_index] != 0 and self.caching_state[self.user_req[user_index][slot_index]-1] != 1:
                    local_task = (1 - self.off_vector[user_index]) * self.task_size[self.user_req[user_index][slot_index]-1]
                    off_task = self.off_vector[user_index] * self.task_size[self.user_req[user_index][slot_index]-1]
                    # 本地计算能耗
                    user_energy_cost += self.user_kappa * self.task_comp_load * local_task * (self.user_fc ** 2)
                    # 卸载传输能耗，user_power单位为mW，故乘以1e-3
                    user_energy_cost += off_task / self.S1_rate_expectation[user_index] * self.user_power * 1e-3
                    # 边缘服务器计算能耗
                    BS_energy_cost += self.server_kappa * self.task_comp_load * off_task * (self.server_fc ** 2)
        # 总能耗成本，包括本地计算能耗、传输能耗、边缘服务器计算能耗
        energy_cost = self.weight_user_ec * user_energy_cost + self.weight_BS_ec * BS_energy_cost
        # 每个用户的平均能耗成本
        per_user_e_cost = energy_cost / self.user_num
        # print("Real energy: ", per_user_e_cost)
        cache_cost = 0.5 * np.sum(np.multiply(self.caching_state, self.result_size))
        total_cost = per_user_e_cost + cache_cost
        total_cost_reward = 1 * total_cost
        return total_cost_reward

    # 最坏成本
    def worst_cost(self):
        '''
        成本最高的情况：所有用户都进行本地卸载 + 缓存容量用满 + 缓存命中率为0
        '''
        local_energy_cost = 0
        for user_index in range(self.user_num):
            for slot_index in range(1, self.s_slot_num+1):
                if self.user_req[user_index][slot_index] != 0:
                    local_energy_cost += self.user_kappa * self.task_comp_load * \
                        self.task_size[self.user_req[user_index][slot_index]-1] * (self.user_fc ** 2)
        # 要得到每个用户的平均能耗，而不是所有用户的总能耗
        per_user_le_cost = local_energy_cost / self.user_num
        # print("Max energy: ", per_user_le_cost)
        # 缓存成本，权重应该稍微小一点
        cache_cost = 0.5 * self.server_c
        worst_cost = per_user_le_cost + cache_cost
        worst_cost_reward = 1 * worst_cost
        return worst_cost_reward

    # 时延违反惩罚
    def latency(self):
        '''
        当时延违反概率超过设定的阈值时，给予惩罚
        注意只有卸载的用户会有时延违反惩罚
        '''
        vio_prob = self.reliability()
        # soft punish
        clip_differences = np.clip(self.probability_bound - vio_prob, -np.inf, 0)
        latency_punish = 200 * np.sum(clip_differences)    # 用总和比较好，反正每个用户都不应该违反
        # hard punish
        # differences = self.probability_bound - vio_prob
        # latency_punish = -100 * np.sum(differences < 0)
        # soft & hard punish
        # clip_differences = np.clip(self.probability_bound - vio_prob, -np.inf, 0)
        # latency_punish = -100 * np.sum(clip_differences < 0) + 1000 * np.sum(clip_differences)
        return latency_punish

    # 时延可靠性
    def reliability(self):
        '''
        计算排队时延超过最大可容忍时延（0.2s）的概率
        '''
        theta_list = []
        prob_list = []

        start_theta = 0.01
        end_theta = 1.0
        theta_step = 0.01

        a = self.task_min   # 均匀分布最小值
        b = self.task_max   # 均匀分布最大值
        hit_rate = []       # 每个用户各自的缓存命中率
        S1_per_slot = self.S1_rate * self.s_slot_tau    # 第一跳瞬时服务量
        S2_per_slot = self.S2_rate * self.s_slot_tau    # 第二跳瞬时服务量

        # 分别计算每个用户各自的缓存命中率
        for user_index in range(self.user_num):
            hit = 0
            not_hit = 0
            for slot_index in range(1, self.s_slot_num+1):
                if self.user_req[user_index][slot_index] != 0:
                    if self.caching_state[self.user_req[user_index][slot_index]-1] == 1:
                        hit += 1
                    else:
                        not_hit += 1
            hit_rate.append(hit / (hit + not_hit))

        for user_index in range(self.user_num):
            THETA = start_theta
            for THETA in np.arange(start_theta, end_theta + 0.001, theta_step):
                # 均匀分布的矩母函数，自己推导
                uniform = np.exp(THETA * self.off_vector[user_index] * a) / (b - a + 1) * (1 - np.exp(THETA * self.off_vector[user_index] * (b - a + 1))) / (1 - np.exp(THETA * self.off_vector[user_index])) * (1 - self.zipf_R[user_index]) * (1 - hit_rate[user_index]) + self.zipf_R[user_index] + (1 - self.zipf_R[user_index]) * hit_rate[user_index]
                K_a = np.log(uniform) / THETA

                K_s1 = np.log(np.sum((1 / self.s_slot_num) * np.exp(-S1_per_slot[user_index] * THETA))) / (-THETA)
                K_s2 = S2_per_slot[user_index]
                
                if K_a > K_s1 or K_a > K_s2:
                    break

            if np.abs(THETA - start_theta) < 1e-9:
                theta = 0.5 * THETA
            else:
                theta = THETA - theta_step
            theta_list.append(theta)
            # 用解出来的θ* (theta)重新计算一次具体值
            uniform = np.exp(theta * self.off_vector[user_index] * a) / (b - a + 1) * (1 - np.exp(theta * self.off_vector[user_index] * (b - a + 1))) / (1 - np.exp(theta * self.off_vector[user_index])) * (1 - self.zipf_R[user_index]) * (1 - hit_rate[user_index]) + self.zipf_R[user_index] + (1 - self.zipf_R[user_index]) * hit_rate[user_index]
            K_a = np.log(uniform) / theta
            prob = np.exp(-self.tolerant_latency * THETA * K_a)
            prob_list.append(0.0 if (np.abs(THETA - end_theta) < 1e-9) else prob)

        return np.array(prob_list)