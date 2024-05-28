import numpy as np

class Environment:
    '''
    This class implements the wireless computing environment
    '''
    def __init__(self, args):
        self.user_num = args.user_num
        self.user_fc = args.user_frequency
        self.user_kappa = args.user_e_coe
        self.user_power = args.user_power
        self.server_fc = args.server_frequency
        self.server_kappa = args.server_e_coe
        self.server_c = args.server_cache_size

        self.task_num = args.task_num
        self.task_comp_load = args.task_computation_load
        self.task_max = args.task_max
        self.task_min = args.task_min
        self.result_max = args.task_result_max
        self.result_min = args.task_result_min

        self.s_slot_tau = args.s_slot_tau
        self.s_slot_num = args.s_slot_num

        self.bandwidth = args.bandwidth
        self.sigma2 = args.sigma2
        self.BS_pos = args.BS_pos

        self.weight_user_ec = args.weight_user_ec
        self.weight_BS_ec = args.weight_BS_ec

        self.action_dim = self.user_num + self.task_num
        self.max_action = args.max_action

        self.probability_bound = args.probability_bound
        self.tolerant_latency = args.tolerant_latency

        self.pathloss_rho = 0.001
        self.pathloss_alpha = 3.5

        self.zipf_N = args.zipf_N
        self.zipf_alpha = np.random.choice([0.7, 0.8], size=self.user_num)
        self.zipf_R = np.random.choice([0.1, 0.2], size=self.user_num)

        self.result_size = np.random.randint(self.result_min, self.result_max + 1, size=self.task_num)

        self.user_req = np.random.randint(0, self.task_num + 1, size=(self.user_num, self.s_slot_num + 1))

        self.BS_position = np.array([self.BS_pos, self.BS_pos])
        self.users_position = np.array(
                    [[np.random.randint(self.BS_pos - 100, self.BS_pos + 101), np.random.randint(self.BS_pos - 100, self.BS_pos + 101)] 
                    for _ in range(self.user_num)])
        self.distances = np.linalg.norm(self.users_position - self.BS_position, axis=1).reshape((-1, 1))

        self.channel_gain_sample()
        
    
    def reset(self):
        self.state = self.user_req[:, -1]
        
        return self.state


    def user_request_transfer(self):
        zipf_sum = np.zeros(self.user_num)
        for i in range(self.task_num):
            for user_index in range(self.user_num):
                zipf_sum[user_index] = zipf_sum[user_index] + 1/((i + 1)**self.zipf_alpha[user_index])
        for user_index in range(self.user_num):
            slot_pre = 0
            self.user_req[user_index][0] = self.user_req[user_index][-1]  # store the request of the last small slot from the previous large slot
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
        The channel gain is sampled for each small time slot
        '''
        gain = np.random.rayleigh(scale=np.sqrt(0.5), size=(self.user_num, self.s_slot_num))
        self.channel_gain = gain * np.sqrt(self.pathloss_rho * np.power(self.distances, -self.pathloss_alpha))


    def step(self, action):
        action = self.preprocess(action)
        self.random_cache()                                                 # Caching policy
        self.off_vector = action[self.task_num:self.user_num+self.task_num] # Offloading policy
        self.server_fc_alloc = action[-self.user_num:]                      # Computing resource allocation policy
        self.user_request_transfer()

        # The task size is sampled for each large time slot
        self.task_size = np.random.randint(self.task_min, self.task_max + 1, size=self.task_num)
        
        # The first hop rate，in Mbit/s
        self.S1_rate = self.bandwidth * np.log2(1 + self.user_power * np.power(self.channel_gain, 2) / self.sigma2)
        # The expected rate of each user in a large time slot
        self.S1_rate_expectation = np.mean(self.S1_rate, axis=1)
        # The second hop rate is constant in a large time slot
        self.S2_rate = self.server_fc_alloc * 1e9 / self.task_comp_load

        cache_reward = self.l_slot_hit_rate() + self.cache_limit()
        fc_reward = self.fc_limit()
        cost_reward = self.worst_cost() - self.total_cost()
        latency_reward = self.latency()
        reward = cache_reward + fc_reward + cost_reward + latency_reward

        self.state = self.user_req[:, -1]

        return reward, self.state, cache_reward, cost_reward, latency_reward
    

    def preprocess(self, action):
        action = np.clip(action, -self.max_action, self.max_action)
        # Caching policy
        action[:self.task_num][action[:self.task_num] >= 0] = 1
        action[:self.task_num][action[:self.task_num] < 0] = 0
        # Offloading policy
        action[self.task_num : self.task_num+self.user_num] = action[self.task_num : self.task_num+self.user_num] * 0.5 + 0.5
        # Computing resource allocation policy
        temp = self.server_fc / self.user_num / 2 * 1e-9
        action[-self.user_num:] = action[-self.user_num:] * temp + temp

        return action


    def random_cache(self):
        self.caching_state = np.zeros(self.task_num)
        remaining_capacity = self.server_c
        for i in range(self.task_num):
            if remaining_capacity <= 0:
                break
            task_idx = np.random.randint(0, self.task_num - 1)
            while self.caching_state[task_idx] == 1:
                task_idx = np.random.randint(0, self.task_num - 1)
            if self.result_size[task_idx] > remaining_capacity:
                break
            self.caching_state[task_idx] = 1
            remaining_capacity -= self.result_size[task_idx]

    
    # Cache hit ratio reward
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


    # Cache overflow penalty
    def cache_limit(self):
        cache_usage = np.sum(np.multiply(self.caching_state, self.result_size))
        cache_punish = 0
        if cache_usage > self.server_c:
            cache_punish = -50 * (cache_usage - self.server_c)

        return cache_punish


    # Computing resource overflow penalty
    def fc_limit(self):
        fc_punish = 10 * np.sum(self.server_fc_alloc)
        fc_punish = 0
        fc_usage = np.sum(self.server_fc_alloc)
        if fc_usage > self.server_fc:
            fc_punish -= 50 * (fc_usage - self.server_fc)

        return fc_punish


    # Total cost
    def total_cost(self):
        '''
        Total cost: Energy cost + Cache cost
        '''
        user_energy_cost = 0
        BS_energy_cost = 0
        for user_index in range(self.user_num):
            for slot_index in range(1, self.s_slot_num+1):
                if self.user_req[user_index][slot_index] != 0 and self.caching_state[self.user_req[user_index][slot_index]-1] != 1:
                    local_task = (1 - self.off_vector[user_index]) * self.task_size[self.user_req[user_index][slot_index]-1]
                    off_task = self.off_vector[user_index] * self.task_size[self.user_req[user_index][slot_index]-1]
                    # Local computing energy consumption
                    user_energy_cost += self.user_kappa * self.task_comp_load * local_task * (self.user_fc ** 2)
                    # Transmission energy consumption
                    user_energy_cost += off_task / self.S1_rate_expectation[user_index] * self.user_power * 1e-3
                    # Edge server computing energy consumption
                    BS_energy_cost += self.server_kappa * self.task_comp_load * off_task * (self.server_fc ** 2)
        energy_cost = self.weight_user_ec * user_energy_cost + self.weight_BS_ec * BS_energy_cost
        per_user_e_cost = energy_cost / self.user_num
        cache_cost = 0.5 * np.sum(np.multiply(self.caching_state, self.result_size))
        total_cost = per_user_e_cost + cache_cost
        total_cost_reward = 1 * total_cost

        return total_cost_reward


    # Worst cost
    def worst_cost(self):
        local_energy_cost = 0
        for user_index in range(self.user_num):
            for slot_index in range(1, self.s_slot_num+1):
                if self.user_req[user_index][slot_index] != 0:
                    local_energy_cost += self.user_kappa * self.task_comp_load * \
                        self.task_size[self.user_req[user_index][slot_index]-1] * (self.user_fc ** 2)
        per_user_le_cost = local_energy_cost / self.user_num
        cache_cost = 0.5 * self.server_c
        worst_cost = per_user_le_cost + cache_cost
        worst_cost_reward = 1 * worst_cost

        return worst_cost_reward


    # Delay violation penalty
    def latency(self):
        '''
        The penalty is given when the delay violation probability exceeds the bound
        '''
        vio_prob = self.reliability()
        # soft punish
        clip_differences = np.clip(self.probability_bound - vio_prob, -np.inf, 0)
        latency_punish = 200 * np.sum(clip_differences) 

        return latency_punish


    # Delay reliability
    def reliability(self):
        '''
        The probability that the actual delay exceeds the maximum tolerable delay
        Using martingale theory
        '''
        theta_list = []
        prob_list = []

        start_theta = 0.01
        end_theta = 1.0
        theta_step = 0.01

        a = self.task_min   # Minimum value of the uniform distribution
        b = self.task_max   # Maximum value of the uniform distribution
        hit_rate = []
        S1_per_slot = self.S1_rate * self.s_slot_tau    # Instantaneous service of the first hop
        S2_per_slot = self.S2_rate * self.s_slot_tau    # Instantaneous service of the second hop

        # Calculate the cache hit ratio for each user separately
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

            uniform = np.exp(theta * self.off_vector[user_index] * a) / (b - a + 1) * (1 - np.exp(theta * self.off_vector[user_index] * (b - a + 1))) / (1 - np.exp(theta * self.off_vector[user_index])) * (1 - self.zipf_R[user_index]) * (1 - hit_rate[user_index]) + self.zipf_R[user_index] + (1 - self.zipf_R[user_index]) * hit_rate[user_index]
            K_a = np.log(uniform) / theta
            prob = np.exp(-self.tolerant_latency * THETA * K_a)
            prob_list.append(0.0 if (np.abs(THETA - end_theta) < 1e-9) else prob)

        return np.array(prob_list)