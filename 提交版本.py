import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
import jieba
from solution.dependency import samples
import random

class DeviceCloudAgent:
    def __init__(self, device_model, cloud_model):
        self.cloud_model = cloud_model
        self.device_model = device_model

    def run(self, query):
        """
        本赛题需要选手构建一个“端云协同决策Agent”。
        根据每道题给定的用户输入在端侧轻量模型和云端大模型之间进行任务分配，以达到尽可能良好的资源利用和用户体验。

        :param query:
        :return:
        """
        # 通过切片操作创建副本并打乱副本的顺序
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        # 定义中文停用词列表
        stop_words = ["的", "了", "是", "我", "你", "他", "它", "这", "那", "和", "与", "呢", "呀", "啊","车载ai","?","？","车载AI","吗","移动","手机",]


        # 自定义文本向量化类，实现fit和transform方法，符合sklearn管道中转换器的要求
        class ChineseTextVectorizer:
            def __init__(self):
                self.vectorizer = TfidfVectorizer(min_df=2, max_df=0.8, ngram_range=(1, 2))  ########  ngram_range=(1, 2) 

            def fit(self, texts, *args, **kwargs):
                texts_tokenized = [" ".join([word for word in jieba.cut(text) if word not in stop_words]) for text in texts]
                self.vectorizer.fit(texts_tokenized)
                return self

            def transform(self, texts):
                texts_tokenized = [" ".join([word for word in jieba.cut(text) if word not in stop_words]) for text in texts]
                return self.vectorizer.transform(texts_tokenized)


        # 提取问题文本和对应的隐私标签（0或1）
        question_texts = [sample[0] for sample in shuffled_samples]
        privacy_labels = np.array([sample[1] for sample in shuffled_samples])

        # 创建文本分类管道，包含自定义的文本向量化类和支持向量机分类器
        text_clf = Pipeline([
            ('vectorizer', ChineseTextVectorizer()),
            ('clf', SVC(kernel='linear'))
        ])

        # 使用数据集训练分类器
        text_clf.fit(question_texts, privacy_labels)


        def is_privacy_related(query):
            """
            判断新问题是否涉及隐私
            """
            prediction = text_clf.predict([query])[0]
            return bool(prediction)

        if bool(prediction):
            final_output=self.device_model.generate(query)
        else:
            #走云侧模型
            final_output=self.cloud_model.generate(query)
        return final_output
