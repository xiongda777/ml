import pandas as pd
import numpy as np

import sklearn
import matplotlib.pyplot as plt
import time
import os
import json
class sklearn_classify():
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
            self.model=RidgeClassifier(**paramters)
        if model_name=="LogisticRegression":
            '''
            #region
            逻辑斯蒂回归:对数线性模型,返回的结果是一个概率值,将样本分类到概率较大所属的类别,一般形式为log(p/(1-p))=w*x =>> p=exp(w*x)/(1+exp(w*x))
            -----------------------------------
            params: penalty:正则化形式 可选["l1","l2","elasticnet","none"] 参数类型:str
                    ---------------------------
                    tol:误差控制项 参数类型:float
                    ----------------------------
                    C:反正则化系数,值越小,正则化能力越强  参数类型:float
                    ---------------------------
                    fit_intercept:截距项 参数类型:bool
                    ------------------------------
                    solver:计算方法,可选["newton-cg","lbfgs","liblinear","sag","saga"] 参数类型:str
                    -----------------------------
                    l1_ratio:l1,l2正则化比率,当选择elasticnet正则化时生效 0 < l1_ratio <1 参数类型:float
                    
            #endregion
            '''
            from sklearn.linear_model import LogisticRegression
            self.model=LogisticRegression(**paramters)
        if model_name=="SVC":
            '''
            #region
            支持向量机,带核函数形式
            -----------------------------------
            params: kernel:核函数 可选["linear","poly","rbf","sigmoid","precomputed"] 参数类型:str
                    ---------------------------
                    degree:poly核函数参数,最大多项式幂 参数类型:int
                    ----------------------------
                    C:反正则化系数,值越小,正则化能力越强  参数类型:float
                    ---------------------------
                    gamma:"rbf","sigmoid","rbf"核系数  可选["scale","auto"] 参数类型:str
                    ------------------------------
                    tol:误差控制项 参数类型:float
                    -----------------------------
                    probability:是否采用概率估计 参数类型:bool
                    
            #endregion
            '''
            from sklearn.svm import SVC
            self.model=SVC(**paramters)
        if model_name=="KNeighborsClassifier":
            '''
            #region
            最近邻分类
            -----------------------------------
            params: n_neighbors:最近邻样本点数量  参数类型:int
                    ---------------------------
                    weights:权重 可选["uniform","distance"]  参数类型:int
                                    uniform:平均权重  distance:根据距离给予权重
                    ----------------------------
                    p:距离度量参数 p=1 曼哈顿距离 p=2欧式距离 对于其他值 使用闵可夫斯基距离  参数类型:int
                    
            #endregion
            '''
            from sklearn.neighbors import KNeighborsClassifier
            self.model=KNeighborsClassifier(**paramters)
        if model_name=="GaussianNB":
            '''
            #region
            朴素贝叶斯分类器,假设p(xi|y)符合高斯分布,p(y)为先验
            -----------------------------------
            params: priors:指定先验  参数类型:list len(list)=n_class
                    ---------------------------
                    
            #endregion
            '''
            from sklearn.naive_bayes import GaussianNB
            self.model=GaussianNB(**paramters)
        if model_name=="MultinomialNB":
            '''
            #region
            朴素贝叶斯分类器,假设p(xi|y)符合多项式分布,p(y)为先验
            -----------------------------------
            params: class_prior:指定先验  参数类型:list len(list)=n_class
                    ---------------------------
                    fit_prior:是否拟合先验 if false p(y)=1/k 参数类型:bool
                    --------------------------
                    alpha:多项式分布的平滑系数 参数类型:float
            #endregion
            '''
            from sklearn.naive_bayes import MultinomialNB
            self.model=MultinomialNB(**paramters)       
        if model_name=="DecisionTreeClassifier":
            '''
            #region
            决策树分类器
            -----------------------------------
            params: criterion:树的划分指标 可选["gini","entropy"]  参数类型:str
                    ---------------------------
                    max_depth:树的最大深度  参数类型:int
                    --------------------------
                    min_samples_leaf:叶子节点包含的最小样本量 参数类型:int
                    --------------------------
                    min_impurity_decrease:如果划分的节点质量提升大于小于该值就进行划分  参数类型:float
            #endregion
            '''
            from sklearn.tree import DecisionTreeClassifier
            self.model=DecisionTreeClassifier(**paramters)       
        elif model_name=="MLPClassifier":
            '''
            #region
            神经网络分类
            ---------------------------------------------
            paramters:  hidden_layer_sizes:隐藏层神经元数量,如(100,) 参数类型:tuple
                        -------------------------
                        activation:激活函数,可选["identity","logistic","tanh","relu"]  参数类型:str
                        --------------------------
                        solver:优化器,可选["lbfgs","sgd","adam"]  参数类型:str
                        --------------------------
                        alpha:L2正则化系数 参数类型:float
                        ---------------------------
                        batch_size:"sgd"优化时使用的最小样本数量 参数类型:int
                        ------------------------------
                        learning_rate:学习率设定方法,可选["constant","invscaling","adaptive"] 参数类型:str
                        ------------------------------
                        learning_rate_init:初始学习率 参数类型:double
                        ------------------------------
                        max_iter:最大迭代次数,每个样本被使用多少次 参数类型:int
                        ------------------------------
                        tol:误差控制项 参数类型:float

            #endregion
            '''
            from sklearn.neural_network import MLPClassifier
            self.model=MLPClassifier(**paramters)
        elif model_name=="BaggingClassifier":
            '''
            #region
            bagging分类
            ---------------------------------------------
            paramters:  base_estimator:基学习器 参数类型:object
                        -------------------------
                        n_estimators:学习器数量 参数类型:int
                        --------------------------
                        max_samples:基学习器训练的样本数量 参数类型:int or float
                        ---------------------------
                        max_features:基学习器训练的特征数量  参数类型:int or float
                        ----------------------------
                        bootstrap:是否有放回的采样 参数类型:bool
                        ----------------------------
                        bootstrap_features:是否有放回的采样特征 参数类型:bool
            #endregion
            '''
            from sklearn.ensemble import BaggingClassifier
            self.model=BaggingClassifier(**paramters)
        elif model_name=="RandomForestClassifier":
            '''
            #region
            随机森林分类
            ---------------------------------------------
            paramters:  n_estimators:学习器数量 参数类型:int
                        -------------------------
                        criterion:评价树的划分指标,可选["gini','entropy'] 参数类型:str
                        ----------------------------
                        bootstrap:是否有放回的采样 参数类型:bool
                        --------------------------
                        max_depth:树的最大深度,如不指定,树会一直分割,导致过拟合  参数类型:int
                        --------------------------
                        min_samples_split:拆分内部节点所需的最小样本数量 参数类型:int or float
                        ---------------------------
                        min_samples_leaf:叶节点所需的最小样本数量 参数类型:int or float
                        ------------------------------
                        min_impurity_decrease:如果继续划分的误差不小于该值,则不会继续划分 参数类型:float

            #endregion
            '''
            from sklearn.ensemble import RandomForestClassifier
            self.model=RandomForestClassifier(**paramters)
        elif model_name=="AdaBoostClassifier":
            '''
            #region
            https://zhuanlan.zhihu.com/p/39972832
            AdaBoost分类,对上一个基模型训练错误的样本给予更高的权重后加入到下一个基学习器,直到达到学习器数量
            ---------------------------------------------
            paramters:  base_estimator:基学习器 参数类型:object
                        -------------------------
                        n_estimators:学习器数量 参数类型:int
                        ----------------------------
                        learning_rate:学习率,权重改变的快慢 参数类型:float
            #endregion
            '''
            from sklearn.ensemble import AdaBoostClassifier
            self.model=AdaBoostClassifier(**paramters) 
        elif model_name=="GradientBoostingClassifier":
            '''
            #region
            https://blog.csdn.net/zhsworld/article/details/102951061
            GradientBoosting分类,基学习器为树模型,下一个基学习器学习上一个学习器预测结果的残差
            ---------------------------------------------
            paramters:  loss:要优化的是损失函数,可选["deviance","exponential",] 参数类型:str
                        -------------------------
                        n_estimators:学习器数量 参数类型:int
                        ----------------------------
                        learning_rate:学习率 参数类型:float
                        --------------------------
                        criterion:划分树时使用的评价函数,可选["friedman_mse","squared_error","mse","mae"] 参数类型:str
                        --------------------------
                        max_depth:树的最大深度,如不指定,树会一直分割,导致过拟合  参数类型:int
                        --------------------------
                        min_samples_split:拆分内部节点所需的最小样本数量 参数类型:int or float
                        ---------------------------
                        min_samples_leaf:叶节点所需的最小样本数量 参数类型:int or float
                        ------------------------------
                        min_impurity_decrease:如果继续划分的误差不小于该值,则不会继续划分 参数类型:float
                        ------------------------------
                        tol:误差控制项 参数类型:float
            #endregion
            '''
            from sklearn.ensemble import GradientBoostingClassifier
            self.model=GradientBoostingClassifier(**paramters)
        elif model_name=="VotingClassifier":
            '''
            #region
            投票分类,多个学习器对同一组样本共同学习,返回它们预测的平均值
            ---------------------------------------------
            paramters:  estimators:学习器 如[("学习器名称1",LinearRegression()),("学习器名称2",RandomForestRegressor())] 参数类型:list
                        -------------------------
                        weights:学习器数量的权重 参数类型:list
                        ------------------------
                        voting:返回类别的方法,可选["hard","soft"] 参数类型:str
            #endregion
            '''
            from sklearn.ensemble import VotingClassifier
            self.model=VotingClassifier(**paramters)    
    
    def params_search(self,params:dict,model,methods:str="Gridcv",score_name:str="accuracy",cv_num:int=5):
        '''
        参数搜索:网格搜索和随机搜索
        ------------------------------
        params: 字典类型 如{'C': [1, 10, 100, 1000], 'kernel': ['linear']} 其键名为模型的参数名,值为要搜索的点
                如 methods为Randomcv, 字典的值可以是一个分布,如高斯分布,均匀分布等
        methods:超参数搜索方法 ["Gridcv","Randomcv"]
        model:要搜索的模型,sklearn模型,可以调用select_model()获得
        score_name:搜索评价指标,可选["accuracy","precision","recall"]
        cv_num:交叉验证次数
        -----------------------------
        return:
            search_result:搜索结果
                cv_results_:搜索过程
                best_estimator_:最好的模型
                best_score_:最高的指标
                best_params_:最好的参数
        '''
        if methods=="Gridcv":
            from sklearn.model_selection import GridSearchCV
            clf = GridSearchCV(estimator=model,param_grid=params,cv=cv_num,scoring=score_name)
            clf.fit(self.x_train,self.y_train)
        elif methods=="Randomcv":
            from sklearn.model_selection import RandomizedSearchCV
            clf = RandomizedSearchCV(estimator=model,param_grid=params,cv=cv_num,scoring=score_name)
            clf.fit(self.x_train,self.y_train)
        search_result={}
        search_result['cv_results_']=clf.cv_results_
        search_result['best_estimator_']=clf.best_estimator_
        search_result['best_score_']=clf.best_score_
        search_result['best_params_']=clf.best_params_
        return search_result

    def get_model_results(self):
        '''
        获取模型在训练集/测试集上的结果
        -----------------------------------------
        return: model_fit_time:模型拟合时间
                model_coef_:模型参数
                max_positive_error:最大正误差
                min_negetive_error:最小负误差
                MAE:平均绝对误差
                MSE：均方误差
                RMS：均方根误差
                R2:决定系数
                error_filename：误差分布直方图路径
                r2_filename:R2图路径
                learning_curve_filename:学习曲线图路径
        --------------------------------------------
        '''
        start=time.time()
        self.model.fit(self.x_train,np.array(self.y_train).ravel())
        end=time.time()
        model_fit_time=end-start
        try:
            model_coef_=self.model.coef_.tolist()
        except AttributeError:
            model_coef_="该模型无法输出参数"

        y_test_pre=self.model.predict(self.x_test)
        #误差记录
        from sklearn.metrics import accuracy_score
        accuracy=accuracy_score(self.y_test,y_test_pre).tolist()#准确率
        from sklearn.metrics import balanced_accuracy_score
        balanced_accuracy=balanced_accuracy_score(self.y_test,y_test_pre)#每个类的平均召回率
        from sklearn.metrics import precision_score
        precision=precision_score(self.y_test,y_test_pre,average=None).tolist()#精确率
        from sklearn.metrics import recall_score
        recall=recall_score(self.y_test,y_test_pre,average=None).tolist()#召回率
        from sklearn.metrics import f1_score
        f1=f1_score(self.y_test,y_test_pre,average=None).tolist()#f1_score
        filepath_1=os.path.dirname(os.path.abspath(__file__))

        #混淆矩阵图
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        cm_filename=os.path.join(filepath_1,'confusion_matrix.png')
        cm = confusion_matrix(self.y_test,y_test_pre)
        cm_display = ConfusionMatrixDisplay(cm,display_labels=self.model.classes_).plot()
        plt.savefig(cm_filename)
        plt.close()

        #roc图
        from sklearn.metrics import roc_curve,auc
        from sklearn.multiclass import OneVsRestClassifier
        # classifier = OneVsRestClassifier(self.model)
        try:
            y_score = self.model.decision_function(self.x_test)
        except AttributeError:
            y_score=self.model.predict_proba(self.x_test)
        roc_filename=os.path.join(filepath_1,'roc.png')
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        from sklearn.preprocessing import label_binarize
        y_test=label_binarize(self.y_test,classes=self.model.classes_)
        for i in range(len(self.model.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(len(self.model.classes_)):
            # RocCurveDisplay(fpr=fpr[i], tpr=tpr[i]).plot()
            plt.plot(fpr[i],tpr[i],label=f"{self.model.classes_[i]}  auc={ roc_auc[i].round(4) }")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(roc_filename)
        plt.close()

        #p_r曲线
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import PrecisionRecallDisplay
        p_r_filename=os.path.join(filepath_1,'p_r.png')
        prec = dict()
        recall_dic = dict()
        for i in range(len(self.model.classes_)):
            prec[i], recall_dic[i], _ = precision_recall_curve(y_test[:,i], y_score[:, i])
        for i in range(len(self.model.classes_)):
            # RocCurveDisplay(fpr=fpr[i], tpr=tpr[i]).plot()
            plt.plot(recall_dic[i], prec[i],label=f"{self.model.classes_[i]}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower right")
        plt.savefig(p_r_filename)
        plt.close()

        return {
            "拟合时间":model_fit_time,
            "模型参数":model_coef_,
            "准确率":accuracy,
            "精确率":precision,
            "召回率":recall,
            "平均召回率":balanced_accuracy,
            "f1_score":f1,
            "混淆矩阵图":cm_filename,
            "roc图":roc_filename,
            "p_r曲线":roc_filename
        }

if __name__=="__main__":
    filepath=r"C:\Users\pc\Desktop\python\数据集\鸢尾花数据集\iris.csv"
    #实例
    s=sklearn_classify(filepath)
    #获取数据
    s.get_file_data()
    #划分数据集,指定x,y列
    s.test_train_split(0.3,[1,2,3,4],[5])
    #选择模型
    s.select_model(model_name='GradientBoostingClassifier',paramters={})
    #参数搜索
    # dic_=s.params_search({"alpha":[0.001,0.01,0.1,0.5,0.8,1,1.5,2,3,5,10]},model=s.select_model(model_name='Ridge',paramters={}))
    #获取结果
    dic=s.get_model_results()
    
    dic=json.dumps(dic,ensure_ascii=False)
    print(dic)