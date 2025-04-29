import time
import numpy as np
import torch
import os
import re
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

client = OpenAI(
    api_key="sk-4d22f3bd581940e0addcd196e22b2c9e", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义辅助模型调用方法
def call_assistant_model(query):
    completion = client.chat.completions.create(
        model="qwen2.5-coder-14b-instruct", 
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': f'{query}是不是车辆端侧模型能处理的问题，回答是或否。'}
        ],
    )
    response = completion.choices[0].message.content.strip()
    return response == "是"

# 定义要忽略的停用词列表，不包括关键代词
stop_words = {
    "的", "请", "吗", "嗯", "呢", "吧","哦", 
}

# 初始化TF-IDF向量化器
vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=100)

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.05):
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
        q_predict = self.Q.get((state, action), 0)
        q_target = q_predict + self.alpha * (reward + self.gamma * max(self.Q.get((next_state, 'device'), 0), self.Q.get((next_state, 'cloud'), 0)) - q_predict)
        self.Q[(state, action)] = q_target

    def save(self, filename):
        q_values = {k: torch.tensor(v) for k, v in self.Q.items()}
        torch.save(q_values, filename)

    def load(self, filename):
        q_values = torch.load(filename, map_location=torch.device('cpu'))
        self.Q = {k: v.item() for k, v in q_values.items()}

class DeviceModel:
    def generate(self, query):
        start_time = time.time()
        result = f"Device model result for {query}"
        end_time = time.time()
        response_time = end_time - start_time
        print(f"Device model response time for '{query}': {response_time:.4f} seconds")
        return result

class CloudModel:
    def generate(self, query):
        start_time = time.time()
        time.sleep(0.1)
        result = f"Cloud model result for {query}"
        end_time = time.time()
        response_time = end_time - start_time
        print(f"Cloud model response time for '{query}': {response_time:.4f} seconds")
        return result

class DeviceCloudAgent:
    def __init__(self, device_model, cloud_model):
        self.cloud_model = cloud_model
        self.device_model = device_model
        self.q_agent = QLearningAgent()
        self.vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=100)
        self.vectorizer.fit(privacy_list + non_privacy_list)

    def preprocess_query(self, query):
        query = query.translate(str.maketrans('', '', string.punctuation))
        query = ' '.join([word for word in query.split() if word not in stop_words])
        query = re.sub(r'\s+', ' ', query).strip()
        return query
    
    def query_to_state(self, query):
        tfidf_features = self.vectorizer.transform([query]).toarray()[0]
        query_length = len(query.split())
        state = np.append(tfidf_features, [query_length])
        return state

    def warm_up(self, combined_list):
        for query, is_private in combined_list:
            state = self.query_to_state(query)
            state_str = str(state)
            action = self.q_agent.choose_action(state_str)
            
            if is_private and action == 'device':
                reward = 10
            elif not is_private and action == 'cloud':
                reward = 10
            else:
                reward = -20
            
            next_state = state_str
            self.q_agent.learn(state_str, action, reward, next_state)

    def train(self, combined_list):
        for query, is_private in combined_list:
            state = self.query_to_state(query)
            state_str = str(state)
            action = self.q_agent.choose_action(state_str)
            result = 'device' if action == 'device' else 'cloud'
            
            if is_private and action == 'device':
                reward = 5
            elif not is_private and action == 'cloud':
                reward = 5
            else:
                reward = -10
            
            next_state = state_str
            self.q_agent.learn(state_str, action, reward, next_state)

    def run(self, query):
        query = self.preprocess_query(query)
        
        assistant_response = call_assistant_model(query)
        
        if assistant_response:
            start_time = time.time()
            result = self.device_model.generate(query)
            end_time = time.time()
            response_time = end_time - start_time
            print(f"Device model response time for '{query}': {response_time:.4f} seconds")
        elif not assistant_response:
            start_time = time.time()
            result = self.cloud_model.generate(query)
            end_time = time.time()
            response_time = end_time - start_time
            print(f"Cloud model response time for '{query}': {response_time:.4f} seconds")
        else:
            state = self.query_to_state(query)
            state_str = str(state)
            action = self.q_agent.choose_action(state_str)
            if action == 'device':
                start_time = time.time()
                result = self.device_model.generate(query)
                end_time = time.time()
                response_time = end_time - start_time
                print(f"Device model response time for '{query}': {response_time:.4f} seconds")
            else:
                start_time = time.time()
                result = self.cloud_model.generate(query)
                end_time = time.time()
                response_time = end_time - start_time
                print(f"Cloud model response time for '{query}': {response_time:.4f} seconds")
        
        return result

def train_and_save_model(agent, combined_list, model_filename):
    agent.train(combined_list)
    model_path = os.path.join('dependency', model_filename)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.q_agent.save(model_path)

def load_and_run_model(agent, model_filename, query):
    model_path = os.path.join('dependency', model_filename)
    agent.q_agent.load(model_path)
    result = agent.run(query)
    return result

# 创建模型实例
device_model = DeviceModel()
cloud_model = CloudModel()

# 创建 DeviceCloudAgent 实例
agent = DeviceCloudAgent(device_model, cloud_model)

# 合并两个列表并随机抽取1000条数据进行预热，同时打乱数据的顺序
combined_list = [(query, True) for query in privacy_list] + [(query, False) for query in non_privacy_list]
random.shuffle(combined_list)
warm_up_list = combined_list[:1000]

# 进行预热
agent.warm_up(warm_up_list)

# 使用上述函数训练模型并保存Q-table
train_and_save_model(agent, combined_list, 'example.pth')

# 使用上述函数加载模型并运行
result = load_and_run_model(agent, 'example.pth', "some query")
print(result)