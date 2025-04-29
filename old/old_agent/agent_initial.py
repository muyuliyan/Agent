import time
import numpy as np
from data import privacy_list, non_privacy_list

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.Q = {}  # Q-table

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(['device', 'cloud'])  # 随机选择
        else:
            return self.Q[state].argmax()  # 选择Q值最大的动作

    def learn(self, state, action, reward, next_state):
        q_predict = self.Q.get((state, action), 0)
        q_target = q_predict + self.alpha * (reward + self.gamma * max(self.Q.get((next_state, 'device'), 0), self.Q.get((next_state, 'cloud'), 0)) - q_predict)
        self.Q[(state, action)] = q_target

class DeviceCloudAgent:
    def __init__(self, device_model, cloud_model):
        self.cloud_model = cloud_model
        self.device_model = device_model
        self.q_agent = QLearningAgent()
        self.privacy_keywords = self.extract_keywords(privacy_list)
        self.non_privacy_keywords = self.extract_keywords(non_privacy_list)

    def extract_keywords(self, data_list):
        return set(word for item in data_list for word in item.split())
    
    def check_privacy(self, query):
        # 检查查询是否涉及隐私
        query_words = set(query.split())
        privacy_match = query_words.intersection(self.privacy_keywords)
        non_privacy_match = query_words.intersection(self.non_privacy_keywords)
        return len(privacy_match) > len(non_privacy_match)

    def run(self, query):
        # 判断是否涉及隐私
        is_private = self.check_privacy(query)
        
        # 使用强化学习模型选择动作
        state = ('query', is_private)  # 状态可以是(query, is_private)
        action = self.q_agent.choose_action(state)
        
        # 根据选择的动作生成结果
        if action == 'device':
            result = self.device_model.generate(query)
        else:
            result = self.cloud_model.generate(query)
        
        # 假设我们根据结果的好坏来给予奖励
        reward = self.calculate_reward(result, action, is_private)
        
        # 更新Q-table
        next_state = ('query', not is_private)  # 假设下一个状态是相反的隐私状态
        self.q_agent.learn(state, action, reward, next_state)
        
        return result

    def calculate_reward(self, result, action, is_private):
        # 根据模型的决策和实际结果来给予奖励
        if action == 'cloud' and is_private:
            return 1  # 云端模型正确处理隐私数据
        elif action == 'device' and not is_private:
            return 1  # 端侧模型正确处理非隐私数据
        else:
            return -1  # 错误的决策

# 假设device_model和cloud_model已经被定义
# agent = DeviceCloudAgent(device_model, cloud_model)
# result = agent.run("some query")