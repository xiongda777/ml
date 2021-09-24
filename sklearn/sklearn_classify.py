import pandas as pd
import numpy as np

import sklearn
import matplotlib.pyplot as plt

class sklean_classify():
    def __init__(self,filepath:str):
        '''
        初始化参数
        ---------------
        filepath:数据文件路径
        '''
        self.filepath=filepath

    def get_file_data(self):
        '''
        读取文件数据 
        ----------------
        return self.dataset:pandas.dataframe格式数据
        '''
        filepath_low=self.filepath.lower()#转换为小写,用于比较后缀名
        last_name=filepath_low.rsplit(".",1)[1]
        if last_name=='txt':
            self.dataset=pd.read_csv(self.filepath,delimiter='\t')
        elif last_name=='csv':
            self.dataset=pd.read_csv(self.filepath)
        elif last_name=='json':
            self.dataset=pd.read_json(self.filepath)
        return self.dataset

    def test_train_split(self,test_rate:float,x_columns:list,y_columns:list):
        '''
        划分训练集,测试集
        ----------------------
        test_rate:测试集比例
        x_columns:自变量所在列
        y_columns:目标变量所在列
        -----------------------
        '''
        self.dataset=self.get_file_data()
        x=self.dataset.iloc[:,x_columns]
        self.label=self.dataset.iloc[:,y_columns]
        from sklearn.preprocessing import StandardScaler
        self.x_scaler=StandardScaler()
        self.x_normalize=self.x_scaler.fit_transform(x)
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test =train_test_split(self.x_normalize, self.label, test_size=test_rate)

    def select_model(self,model_name:str,paramters:dict={}):
        '''
        选择模型,输入参数
        -----------------
        model_name:sklearn库模型的名称
        paramters:对应的模型参数
        ---------------------------
        '''
        if model_name=="RidgeClassifier":
            '''
            #region
            岭分类器:对于二分类任务,对label编码为{-1,1},学习方法和回归相同,预测值为正则为正类
                    对于多分类任务,对label采用one hot编码{[0,1],[1,0]},预测为多输出,预测类为两个输出中较大的值对应的类
            -----------------------------------
            params: alpha:正则化系数 参数类型:float
                    ---------------------------
                    fit_intercept:截距项 参数类型:bool
                    ------------------------------
                    solver:计算方法,可选["auto","svd","cholesky","lsqr","sparse_cg","sag","saga"] 参数类型:str
            #endregion
            '''
            from sklearn.linear_model import RidgeClassifier
            self.model=RidgeClassifier()
        