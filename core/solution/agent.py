import json

# 模型类
class Model:
    def __init__(self, name, max_complexity, latency):
        self.name = name
        self.max_complexity = max_complexity
        self.latency = latency

# 任务复杂性评估函数
def evaluate_task_complexity(question):
    # 简单地使用问题的长度作为复杂性的指标
    return len(question.split())

# 决策函数
def decide_model(question, edge_model, cloud_model):
    task_complexity = evaluate_task_complexity(question)
    if task_complexity <= edge_model.max_complexity:
        return edge_model
    else:
        return cloud_model

# 执行任务
def execute_task(question, selected_model):
    print(f"Task: {question}")
    print(f"Selected Model: {selected_model.name}")
    print(f"Expected Latency: {selected_model.latency} seconds")
    # 这里可以添加调用模型的代码
    answer = f"Answer from {selected_model.name}: {question}"
    print(f"Answer: {answer}")

# 实例化模型
edge_model = Model("llama3-8B-Instruct", max_complexity=10, latency=0.1)
cloud_model = Model("Qwen12-72B-Instruct", max_complexity=50, latency=1.0)

# 读取和处理JSON文件
def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for session in data['sessions']:
            for turn_id, turn in session['chatTurns'].items():
                question = turn['question']
                selected_model = decide_model(question, edge_model, cloud_model)
                execute_task(question, selected_model)

# 文件路径
file_path = 'test_questions.json'

# 处理JSON文件
process_json_file("test_questions.json")