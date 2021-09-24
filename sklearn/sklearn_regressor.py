import pandas as pd
import numpy as np

import sklearn
import matplotlib.pyplot as plt

import json
import os
import time
import logging

font={
    'family':'STSONG'
}
plt.rc("font",**font)
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.DEBUG #设置日志输出格式
                    ,filename="demo.log" #log日志输出的文件位置和文件名
                    ,filemode="w" #文件的写入格式，w为重新写入文件，默认是追加
                    ,format="%(asctime)s - %(levelname)-9s: %(message)s" #日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
                    )
logger = logging.getLogger(__name__)

class sklearn_regressor():
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
        y=self.dataset.iloc[:,y_columns]
        from sklearn.preprocessing import StandardScaler
        self.x_scaler=StandardScaler()
        self.x_normalize=self.x_scaler.fit_transform(x)
        self.y_scaler=StandardScaler()
        self.y_normalize=self.y_scaler.fit_transform(y)
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test =train_test_split(self.x_normalize, self.y_normalize, test_size=test_rate)

    def select_model(self,model_name:str,paramters:dict={}):
        '''
        选择模型,输入参数
        -----------------
        model_name:sklearn库模型的名称
        paramters:对应的模型参数
        ---------------------------
        '''
        if model_name=="LinearRegression":
            '''
            #region
            简单线性模型
            ------------------
            paramters:  fit_intercept:截距项 参数类型:bool
            #endregion
            '''
            from sklearn.linear_model import LinearRegression
            self.model=LinearRegression(**paramters)
        elif model_name=="Ridge":
            '''
            #region
            带L2正则化的线性模型
            -------------------
            paramters:  alpha:L2正则化系数 参数类型:list 
                        多目标回归可设置不同alpha,eg:alpha=[0.1,0.2]
                        ---------
                        fit_intercept:截距项 参数类型:bool
                        ----------------
                        tol:误差控制项  参数类型:float
            #endregion
            '''
            from sklearn.linear_model import Ridge
            self.model=Ridge(**paramters)
        elif model_name=="Lasso":
            '''
            #region
            带L1正则化的线性模型,产生一个稀疏的模型
            ------------------------------------
            paramters:  alpha:L1正则化系数 参数类型:float
                        Lasso多目标回归不可设置不同alpha
                        ------------------
                        fit_intercept:截距项 参数类型:bool
                        ------------------
                        tol:误差控制项  参数类型:float
            #endregion
            '''
            from sklearn.linear_model import Lasso
            self.model=Lasso(**paramters)
        elif model_name=="ElasticNet":
            '''
            #region
            同时带L1,L2正则化的线性模型
            ------------------------------------
            paramters:  alpha:正则化系数  参数类型:float
                        ------------------
                        l1_ratio:L1/L2比值  参数类型:float
                        --------------------------------
                        fit_intercept:截距项 参数类型:bool
                        ------------------
                        tol:误差控制项  参数类型:float
            #endregion
            '''
            from sklearn.linear_model import ElasticNet
            self.model=ElasticNet(**paramters)
        elif model_name=="Lars":
            '''
            #region
            Least-angle regression:逐步回归的改进模型,效率更高,具有特征选择的功能
            ------------------------------------
            paramters:  alpha:正则化系数  参数类型:float
                        ------------------
                        l1_ratio:L1/L2比值  参数类型:float
                        --------------------------------
                        fit_intercept:截距项 参数类型:bool
                        ------------------
                        tol:误差控制项  参数类型:float
            #endregion
            '''
            from sklearn.linear_model import ElasticNet
            self.model=ElasticNet(**paramters)
        elif model_name=="BayesianRidge":
            '''
            #region
            https://zhuanlan.zhihu.com/p/403618259
            贝叶斯线性回归的最大对数后验 等价于  最小化平方损失+L2正则化
            拟合的结果是最大化后验概率下的模型参数
            -------------------------------------------------------
            paramters:  alpha_1:数据gamma分布的形状参数 参数类型:float
                        --------------------------------------
                        alpha_2:数据gamma分布的逆尺度参数 参数类型:float
                        --------------------------------------
                        lambda_1:模型系数gamma分布的形状参数 参数类型:float
                        ---------------------------------------
                        lambda_2:模型系数gamma分布的形状参数 参数类型:float
                        -------------------------------------
                        fit_intercept:截距项 参数类型:bool
                        --------------------------------------
                        tol:误差控制项  参数类型:float
            #endregion
            '''
            from sklearn.linear_model import BayesianRidge
            self.model=BayesianRidge(**paramters)
        elif model_name=="KernelRidge":
            '''
            #region
            https://zhuanlan.zhihu.com/p/72517223
            利用核方法的岭回归
            -----------------------------------
            paramters:  alpha:正则化系数 参数类型:float
                        -----------------------------
                        kernel:核函数名称 可选["additive_chi2","chi2","linear","laplacian","polynomial","rbf","sigmoid"] 参数类型:str
                        ------------------------------
                        gamma:RBF, laplacian, polynomial, exponential chi2 and sigmoid kernels核函数参数  参数类型:float
                        ------------------------------
                        degree:多项式核参数  参数类型:float
            #endregion
            '''
            from sklearn.kernel_ridge import KernelRidge
            self.model=KernelRidge(**paramters)
        elif model_name=="SVR":
            '''
            #region
            https://zhuanlan.zhihu.com/p/72517223
            利用核方法的支持向量机回归
            -----------------------------------
            paramters:  alpha:正则化系数 参数类型:float
                        -----------------------------
                        kernel:核函数名称 可选["linear","poly","rbf","sigmoid"] 参数类型:str
                        ------------------------------
                        gamma:RBF, laplacian, poly,sigmoid kernels核函数参数 可选["scale","auto"] 参数类型:str
                        参数取值定义:https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR
                        ------------------------------
                        degree:多项式核参数  参数类型:float
                        --------------------------------------
                        tol:误差控制项  参数类型:float
                        --------------------------------------
                        C:支持向量机正则化系数 参数类型:float
                        -------------------------------------
                        epsilon:svr回归模型中对样本的误差小于epsilon时,不计算在损失函数中  参数类型:float
            #endregion
            '''
            from sklearn.svm import SVR
            self.model=SVR(**paramters)
        elif model_name=="SGDRegressor":
            '''
            #region
            batch GD,mini-batch GD,SGD:https://zhuanlan.zhihu.com/p/357963858
            使用随机梯度下降法优化的线性模型,适用于大规模数据集
            -----------------------------------
            paramters:  loss:损失函数类型 可选["squared_loss","huber","epsilon_insensitive"] 参数类型:str
                        -----------------------------------
                        penalty:正则化系数类型 可选["l1","l2","elasticnet"]  参数类型:sr
                        ------------------------------------
                        alpha:正则化系数 参数类型:float
                        -----------------------------------
                        l1_ratio:elasticnet中l1与l2正则化系数的比值 取值范围[0,1] 参数类型:float
                        --------------------------------------------------------------------
                        fit_intercept:截距项 参数类型:bool
                        ------------------------------
                        tol:误差控制项  参数类型:float
                        ------------------------------
                        max_iter:最大迭代次数 参数类型:int
                        --------------------------------------
                        learning_rate:学习率类型 可选["constant","optimal","invscaling","adaptive"] 参数类型:str
                        --------------------------------------
                        eta0:初始学习率 参数类型:float
                        -------------------------------------
                        epsilon:svr回归模型中对样本的误差小于epsilon时,不计算在损失函数中,loss为epsilon_insensitive时生效  参数类型:float
            #endregion
            '''
            from sklearn.linear_model import SGDRegressor
            self.model=SGDRegressor(**paramters)
        elif model_name=="KNeighborsRegressor":
            '''
            #region
            最近邻回归,用最相近的点的值来拟合位置样本的值
            -----------------------------------
            paramters:  n_neighbors:最近邻的样本数量 参数类型:int
                        ----------------------------------------
                        weights:权重 可选["uniform","distance"] 参数类型:str
                                uniform:所有点在回归时同等权重
                                distance:距离更近的点具有更高的权重
                        ----------------------------------------
                        algorithm:搜索方法 可选["auto","ball_tree","kd_tree","brute"] 参数类型:str
                        -----------------------------------------
                        leaf_size:构造树的节点数量 参数类型:int
                        -----------------------------------------
                        p:距离度量 参数类型:int
                        -----------------------------------------
            #endregion
            '''
            from sklearn.neighbors import KNeighborsRegressor
            self.model=KNeighborsRegressor(**paramters)
        elif model_name=="GaussianProcessRegressor":
            '''
            #region
            https://zhuanlan.zhihu.com/p/350389546
            高斯过程回归:贝叶斯线性回归+加核函数,解决高维度问题
            ---------------------------------------------
            paramters:  kernel:高斯过程的协方差函数
                        -------------------------
                        alpha:观察过程中的噪声方差,可对应于L2正则化参数 参数类型:float
            #endregion
            '''
            from sklearn.gaussian_process import GaussianProcessRegressor
            self.model=GaussianProcessRegressor(**paramters)
        elif model_name=="PLSRegression":
            '''
            #region
            偏最小二乘回归:将高维度特征降低到低纬度空间,寻找X-Y最大相关性的降维回归方法
            ---------------------------------------------
            paramters:  n_components:降维后的主成分 参数类型:int
                        -------------------------
                        tol:误差控制项  参数类型:float
            #endregion
            '''
            from sklearn.cross_decomposition import PLSRegression
            self.model=PLSRegression(**paramters)
        elif model_name=="DecisionTreeRegressor":
            '''
            #region
            cart回归树
            ---------------------------------------------
            paramters:  criterion:用于评价树划分质量的指标,可选["mse","friedman_mse","mae","poisson"] 参数类型:str
                        -------------------------
                        splitter:每个节点的划分策略,可选["best","random"]  参数类型:str
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
            from sklearn.tree import DecisionTreeRegressor
            self.model=DecisionTreeRegressor(**paramters)
        elif model_name=="MLPRegressor":
            '''
            #region
            神经网络回归
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
            from sklearn.neural_network import MLPRegressor
            self.model=MLPRegressor(**paramters)
        elif model_name=="BaggingRegressor":
            '''
            #region
            bagging回归
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
            from sklearn.ensemble import BaggingRegressor
            self.model=BaggingRegressor(**paramters)
        elif model_name=="RandomForestRegressor":
            '''
            #region
            随机森林回归
            ---------------------------------------------
            paramters:  n_estimators:学习器数量 参数类型:int
                        -------------------------
                        criterion:评价树的划分指标,可选["mse','mae'] 参数类型:str
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
            from sklearn.ensemble import RandomForestRegressor
            self.model=RandomForestRegressor(**paramters)
        elif model_name=="AdaBoostRegressor":
            '''
            #region
            https://zhuanlan.zhihu.com/p/39972832
            AdaBoost回归,对上一个基模型训练错误的样本给予更高的权重后加入到下一个基学习器,直到达到学习器数量
            ---------------------------------------------
            paramters:  base_estimator:基学习器 参数类型:object
                        -------------------------
                        n_estimators:学习器数量 参数类型:int
                        ----------------------------
                        learning_rate:学习率,权重改变的快慢 参数类型:float
                        --------------------------
                        loss:更新权重时使用的误差函数,可选["linear","square","exponential"] 参数类型:str
            #endregion
            '''
            from sklearn.ensemble import AdaBoostRegressor
            self.model=AdaBoostRegressor(**paramters)   
        elif model_name=="GradientBoostingRegressor":
            '''
            #region
            https://blog.csdn.net/zhsworld/article/details/102951061
            GradientBoosting回归,基学习器为树模型,下一个基学习器学习上一个学习器预测结果的残差
            ---------------------------------------------
            paramters:  loss:要优化的是损失函数,可选["ls","lad","huber","quantile"] 参数类型:str
                        -------------------------
                        n_estimators:学习器数量 参数类型:int
                        ----------------------------
                        learning_rate:学习率 参数类型:float
                        --------------------------
                        criterion:划分树时使用的评价函数,可选["linear","square","exponential"] 参数类型:str
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
            from sklearn.ensemble import GradientBoostingRegressor
            self.model=GradientBoostingRegressor(**paramters)  
        elif model_name=="VotingRegressor":
            '''
            #region
            VotingRegressor回归,多个学习器对同一组样本共同学习,返回它们预测的平均值
            ---------------------------------------------
            paramters:  estimators:学习器 如[("学习器名称1",LinearRegression()),("学习器名称2",RandomForestRegressor())] 参数类型:list
                        -------------------------
                        weights:学习器数量的权重 参数类型:list
            #endregion
            '''
            from sklearn.ensemble import VotingRegressor
            self.model=VotingRegressor(**paramters) 
        elif model_name=="StackingRegressor":
            '''
            #region
            https://www.cnblogs.com/Christina-Notebook/p/10063146.html
            StackingRegressor,将基模型在训练集和测试集的预测值做为元模型的特征,在元模型上得出结果
            ---------------------------------------------
            paramters:  estimators:基模型 如[("学习器名称1",LinearRegression()),("学习器名称2",RandomForestRegressor())] 参数类型:list
                        -------------------------
                        final_estimator:元模型 参数类型:object
                        --------------------------
                        cv:交叉验证数目 参数类型:int
            #endregion
            '''
            from sklearn.ensemble import StackingRegressor
            self.model=StackingRegressor(**paramters) 
        return self.model

    def params_search(self,params:dict,model,methods:str="Gridcv",score_name:str="neg_mean_squared_error",cv_num:int=5):
        '''
        参数搜索:网格搜索和随机搜索
        ------------------------------
        params: 字典类型 如{'C': [1, 10, 100, 1000], 'kernel': ['linear']} 其键名为模型的参数名,值为要搜索的点
                如 methods为Randomcv, 字典的值可以是一个分布,如高斯分布,均匀分布等
        methods:超参数搜索方法 ["Gridcv","Randomcv"]
        model:要搜索的模型,sklearn模型,可以调用select_model()获得
        score_name:搜索评价指标,可选["neg_mean_absolute_error","neg_mean_squared_error","r2","max_error"]
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
        if score_name=="neg_mean_absolute_error" or score_name=="neg_mean_absolute_error":
            search_result['best_score_']=-clf.best_score_
        else:
            search_result['best_score_']=-clf.best_score_
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
        self.model.fit(self.x_train,self.y_train)
        end=time.time()
        model_fit_time=end-start
        try:
            model_coef_=self.model.coef_.tolist()
        except AttributeError:
            model_coef_="该模型无法输出参数"

        y_test_pre=self.model.predict(self.x_test)
        y_test_pre=self.y_scaler.inverse_transform(np.array(y_test_pre)).ravel()
        self.y_test=self.y_scaler.inverse_transform(self.y_test).ravel()
        #误差记录
        ERROR=self.y_test-y_test_pre
        max_positive_error=max(ERROR)#最大正误差
        min_negetive_error=min(ERROR)#最小负误差
        MAE=sklearn.metrics.mean_absolute_error(self.y_test,y_test_pre)#平均绝对误差
        MSE=sklearn.metrics.mean_squared_error(self.y_test,y_test_pre)#均方误差
        RMS=np.sqrt(MSE)#均方根误差
        from sklearn.metrics import r2_score
        R2=round(r2_score(self.y_test,y_test_pre),3)#r2_score
        filepath_1=os.path.dirname(os.path.abspath(__file__))

        #误差分布直方图
        error_filename=os.path.join(filepath_1,'error.png')
        error=[abs(x) for x in ERROR]
        plt.hist(error)
        plt.savefig(error_filename)
        plt.close()

        #R2图
        r2_filename=os.path.join(filepath_1,'r2.png')
        plt.plot(self.y_test,self.y_test,color='b')
        plt.scatter(self.y_test,y_test_pre,color='r')
        plt.legend(title=f"r2_score={R2}")
        plt.savefig(r2_filename)
        plt.close()

        #学习曲线
        learning_curve_filename=os.path.join(filepath_1,'learning_curve.png')
        train_sizes, train_scores, valid_scores = sklearn.model_selection.learning_curve(
            self.model,self.x_normalize, self.y_normalize,train_sizes=[0.1,0.3,0.5,0.7,0.9],scoring="neg_median_absolute_error")  
        train_scores_mean=np.mean(train_scores,axis=1)
        valid_scores_mean=np.mean(valid_scores,axis=1)
        plt.plot(train_sizes,train_scores_mean, color="r",label='训练曲线')
        plt.plot(train_sizes,valid_scores_mean, color="g",label='验证曲线')
        plt.legend()
        plt.savefig(learning_curve_filename)
        plt.close()

        return {
            "拟合时间":model_fit_time,
            "模型参数":model_coef_,
            "最大正误差":max_positive_error,
            "最小负误差":min_negetive_error,
            "平均绝对误差":MAE,
            "均方误差":MSE,
            "均方根误差":RMS,
            "误差分布直方图":error_filename,
            "R2 图":r2_filename,
            "学习曲线":learning_curve_filename
        }

if __name__=="__main__":
    filepath=r"C:\Users\pc\Desktop\test.json"
    #实例
    s=sklearn_regressor(filepath)
    #获取数据
    s.get_file_data()
    #划分数据集,指定x,y列
    s.test_train_split(0.3,[1,2,4,5],[3])
    #选择模型
    s.select_model(model_name='Ridge',paramters={})
    #参数搜索
    dic_=s.params_search({"alpha":[0.001,0.01,0.1,0.5,0.8,1,1.5,2,3,5,10]},model=s.select_model(model_name='Ridge',paramters={}))
    #获取结果
    dic=s.get_model_results()
    
    dic=json.dumps(dic,ensure_ascii=False)

   
