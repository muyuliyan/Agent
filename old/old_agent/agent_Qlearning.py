import data
import time
import numpy as np
import torch
import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# 定义要忽略的停用词列表，不包括关键代词
stop_words = {
    "的", "是", "在", "和", "了", "请", "吗", "嗯", "呢", "我们", "你们", "他们", "一些"
    "有", "能", "会", "可以", "应该", "必须", "能够", "会", "可以", "要", "需要", "吧", "呢", "哦", "推荐", "如何", "车辆"
}

# 定义实时信息查询关键词列表
real_time_keywords = ["实时", "最新", "当前", "动态", "今天", "如今", "当下", "当今", "最近", "天气"]

# 初始化TF-IDF向量化器
vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=100)

def is_real_time_query(query):
    # 定义实时信息查询的判断逻辑，待优化
    query = re.sub(r'\s+', ' ', query).strip().split()
    return any(keyword in query for keyword in real_time_keywords)

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
        start_time = time.time()  # 开始时间
        time.sleep(0.1)  # 模拟一些计算或延时
        result = f"Device model result for {query}"
        end_time = time.time()  # 结束时间
        response_time = end_time - start_time  # 响应时间
        print(f"Device model response time for '{query}': {response_time:.4f} seconds")
        return result

class CloudModel:
    def generate(self, query):
        # 这里是云端模型处理查询的逻辑
        start_time = time.time()  # 开始时间
        time.sleep(0.1)  # 模拟一些计算或延时
        result = f"Cloud model result for {query}"
        end_time = time.time()  # 结束时间
        response_time = end_time - start_time  # 响应时间
        print(f"Cloud model response time for '{query}': {response_time:.4f} seconds")
        return result
    
class DeviceCloudAgent:
    def __init__(self, device_model, cloud_model):
        self.cloud_model = cloud_model
        self.device_model = device_model
        self.q_agent = QLearningAgent()
        self.vectorizer = vectorizer
        self.vectorizer.fit(privacy_list + non_privacy_list)

    def preprocess_query(self, query):
        # 移除标点符号
        query = query.translate(str.maketrans('', '', string.punctuation))
        # 移除特定语气词和停用词，保留关键代词
        query = ' '.join([word for word in query.split() if word not in stop_words])
        # 处理多个空格
        query = re.sub(r'\s+', ' ', query).strip()
        return query
    
    def query_to_state(self, query):
        # 将查询转换为TF-IDF特征向量
        tfidf_features = self.vectorizer.transform([query]).toarray()[0]
        # 检查查询是否为实时信息查询
        is_real_time = is_real_time_query(query)
        # 查询的长度
        query_length = len(query.split())
        # 关键词的频率
        keyword_frequency = sum(1 for word in query.split() if word in real_time_keywords)
        # 将实时信息查询标志、查询长度和关键词频率加入到状态表示中
        state = np.append(tfidf_features, [int(is_real_time), query_length, keyword_frequency])
        return state

    def train(self, privacy_list, non_privacy_list):
        for query, is_private in zip(privacy_list + non_privacy_list, [True] * len(privacy_list) + [False] * len(non_privacy_list)):
            state = self.query_to_state(query)  # 状态是查询的TF-IDF特征向量和实时信息标志
            state_str = str(state)  # 将状态转换为字符串
            action = self.q_agent.choose_action(state_str)  # 选择动作
            result = 'device' if action == 'device' else 'cloud'
            
            # 给予奖励
            if is_private and action == 'device':
                reward = 5  # 遇到隐私问题，选择端侧模型，给予奖励
            elif not is_private and action == 'cloud':
                reward = 5  # 遇到非隐私问题，选择云端模型，给予奖励
            else:
                reward = -10  # 其他情况，给予惩罚
            
            # 更新Q-table
            next_state = state_str  # 假设下一个状态是相同的查询内容
            self.q_agent.learn(state_str, action, reward, next_state)

    def run(self, query, is_private=None):
        # 使用强化学习模型选择动作
        state = self.query_to_state(query)  # 状态是预处理后的查询内容
        state_str = str(state)  # 将状态转换为字符串
        
        # 检查查询是否为实时信息查询
        if is_real_time_query(query):
            # 实时信息查询，直接选择云端模型处理
            start_time = time.time()  # 开始时间
            result = self.cloud_model.generate(query)
            end_time = time.time()  # 结束时间
            response_time = end_time - start_time  # 响应时间
            print(f"Real-time query response time for '{query}': {response_time:.4f} seconds")
            return result
        
        else:
            # 使用强化学习模型选择动作
            action = self.q_agent.choose_action(state_str)

        # 根据选择的动作生成结果
        if action == 'device':
            start_time = time.time()  # 开始时间
            result = self.device_model.generate(query)
            end_time = time.time()  # 结束时间
            response_time = end_time - start_time  # 响应时间
            print(f"Device model response time for '{query}': {response_time:.4f} seconds")
        elif action == 'cloud':
            start_time = time.time()  # 开始时间
            result = self.cloud_model.generate(query)
            end_time = time.time()  # 结束时间
            response_time = end_time - start_time  # 响应时间
            print(f"Cloud model response time for '{query}': {response_time:.4f} seconds")
        else:
            result = "无法确定隐私状态，无法处理查询"
        
        # 如果是预热阶段，更新Q-table
        if is_private is not None:
            # 给予奖励
            if is_private and action == 'device':
                reward = 5  # 遇到隐私问题，选择端侧模型，给予奖励
            elif not is_private and action == 'cloud':
                reward = 5  # 遇到非隐私问题，选择云端模型，给予奖励
            else:
                reward = -10  # 其他情况，给予惩罚
            
            # 更新Q-table
            next_state = state_str  # 假设下一个状态是相同的查询内容
            self.q_agent.learn(state_str, action, reward, next_state)

        return result

def train_and_save_model(agent, privacy_data, non_privacy_data, model_filename):
    # 训练模型
    agent.train(privacy_data, non_privacy_data)
    
    # 保存Q-table到指定路径
    model_path = os.path.join('dependency', model_filename)
    agent.q_agent.save(model_path)

def load_and_run_model(agent, model_filename, query):
    # 加载Q-table从指定路径
    model_path = os.path.join('dependency', model_filename)
    agent.q_agent.load(model_path)
    
    # 预热模型，先跑一遍已知的问题
    for query, is_private in zip(privacy_list + non_privacy_list, [True] * len(privacy_list) + [False] * len(non_privacy_list)):
        agent.run(query, is_private)
    
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