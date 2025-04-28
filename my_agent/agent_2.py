import time
import numpy as np
import torch
from dependency.data import privacy_list, non_privacy_list

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
            # 选择Q值最大的动作
            actions = ['device', 'cloud']
            return max(actions, key=lambda x: self.Q.get((state, x), 0))

    def learn(self, state, action, reward, next_state):
        if (state, action) in self.Q:
            q_predict = self.Q[(state, action)]
        else:
            q_predict = 0
        q_target = q_predict + self.alpha * (reward + self.gamma * max(self.Q.get((next_state, 'device'), 0), self.Q.get((next_state, 'cloud'), 0)) - q_predict)
        self.Q[(state, action)] = q_target

    def save(self, filename):
        # 将 Q-table 转换为 PyTorch 张量
        q_values = {k: torch.tensor(v) for k, v in self.Q.items()}
        torch.save(q_values, filename)

    def load(self, filename):
        # 加载 Q-table
        q_values = torch.load(filename, map_location=torch.device('cpu'))
        # 将张量转换回 Q-table 字典
        self.Q = {k: v.item() for k, v in q_values.items()}

class DeviceModel:
    def generate(self, query):
        # 这里是设备端模型处理查询的逻辑
        return f"Device model result for {query}"

class CloudModel:
    def generate(self, query):
        # 这里是云端模型处理查询的逻辑
        return f"Cloud model result for {query}"

class DeviceCloudAgent:
    def __init__(self, device_model, cloud_model):
        self.cloud_model = cloud_model
        self.device_model = device_model
        self.q_agent = QLearningAgent()

    def train(self, privacy_list, non_privacy_list):
        for query, is_private in zip(privacy_list + non_privacy_list, [True] * len(privacy_list) + [False] * len(non_privacy_list)):
            state = query  # 状态是查询内容
            action = self.q_agent.choose_action(state)  # 选择动作
            result = 'device' if action == 'device' else 'cloud'
            
            # 给予奖励
            if is_private and action == 'device':
                reward = 5  # 遇到隐私问题，选择端侧模型，给予奖励
            elif not is_private and action == 'cloud':
                reward = 5  # 遇到非隐私问题，选择云端模型，给予奖励
            else:
                reward = -50  # 其他情况，给予惩罚
            
            # 更新Q-table
            next_state = query  # 假设下一个状态是相同的查询内容
            self.q_agent.learn(state, action, reward, next_state)

    def run(self, query):
        # 使用强化学习模型选择动作
        state = query  # 状态是查询内容
        action = self.q_agent.choose_action(state)  # 总是使用强化学习选择动作
        
        # 根据选择的动作生成结果
        if action == 'device':
            result = self.device_model.generate(query)
        elif action == 'cloud':
            result = self.cloud_model.generate(query)
        else:
            result = "无法确定隐私状态，无法处理查询"
        
        return result

def train_and_save_model(agent, privacy_data, non_privacy_data, model_filename):
    # 训练模型
    agent.train(privacy_data, non_privacy_data)
    
    # 保存Q-table
    agent.q_agent.save(model_filename)

def load_and_run_model(agent, model_filename, query):
    # 加载Q-table
    agent.q_agent.load(model_filename)
    
    # 运行模型
    result = agent.run(query)
    return result

# 创建模型实例
device_model = DeviceModel()
cloud_model = CloudModel()

# 创建 DeviceCloudAgent 实例
agent = DeviceCloudAgent(device_model, cloud_model)

# 使用上述函数训练模型并保存Q-table
train_and_save_model(agent, privacy_list, non_privacy_list, 'example.pth')

# 使用上述函数加载模型并运行
result = load_and_run_model(agent, 'example.pth', "some query")
print(result)

# 假设device_model和cloud_model已经被定义
# agent = DeviceCloudAgent(device_model, cloud_model)
# agent.train(privacy_list, non_privacy_list)  # 训练模型
# result = agent.run("some query")  # 运行模型 

# 使用上述函数训练模型并保存Q-table
# train_and_save_model(agent, privacy_list, non_privacy_list, 'q_table.pth')

# 使用上述函数加载模型并运行
# result = load_and_run_model(agent, 'q_table.pth', "some query")
# print(result)