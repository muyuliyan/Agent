
import random

class DecisionAgent:
    def __init__(self):
        # 初始化决策代理
        pass

    def decide_task_location(self, task_data):
        """
        根据任务数据决定任务是在云端执行还是在终端侧执行。
        这里我们简单地根据任务数据的某个特征来决定。
        """
        # 假设任务数据是一个数字，如果数字大于50，则在云端执行，否则在终端侧执行
        if task_data > 50:
            return "cloud"
        else:
            return "edge"

    def execute_task(self, task_data):
        """
        执行任务，根据决策结果调用不同的函数。
        """
        location = self.decide_task_location(task_data)
        if location == "cloud":
            self.execute_cloud_task(task_data)
        else:
            self.execute_edge_task(task_data)

    def execute_cloud_task(self, task_data):
        """
        执行云端任务。
        """
        print(f"Executing cloud task with data: {task_data}")

    def execute_edge_task(self, task_data):
        """
        执行终端侧任务。
        """
        print(f"Executing edge task with data: {task_data}")

# 创建决策代理实例
agent = DecisionAgent()

# 模拟一些任务数据
task_data_list = [random.randint(0, 100) for _ in range(10)]

# 对每个任务数据执行任务
for task_data in task_data_list:
    agent.execute_task(task_data)