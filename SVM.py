'''
SVM classifier
'''
import numpy as np
import cvxopt.solvers
import logging

# 用于限制某些αi的值过小
_MIN_Support_Multiplier = 1e-5
class SVM():
    def __init__(self,kernel,c):
        '''
        :param kernel:核函数
        :param c: 松弛变量，越大越接近理想SVM分类器
        '''
        self._kernel = kernel
        self._c = c
    def _kernel_(self,X):
        '''
        :param X:输入的数据为n*d的矩阵
        :return:计算后的核函数值矩阵n*n的矩阵，其中存放的是实数值，这说明X之间的乘法是X^T*X
        '''
        sample_num = X.shape[0]
        self._kernel_matrix = np.zeros([sample_num,sample_num])
        for i in range(sample_num):
            for j in range(sample_num):
                self._kernel_matrix[i,j] = self._kernel(X[i,:],X[j,:])
    def _fit_(self,X,Y):
        '''
        :param X:输入数据，n*d的矩阵
        :param Y: 数据的标签，n*1的向量
        :return:最优化之后的α值的序列
        '''
        #获取输入数据X的数据项的数目，以及每个数据项的维度大小。
        sample_num,feature_num = X.shape
        kernel_matrix = self._kernel_(X)
        #p对应于αixiyi
        p = cvxopt.matrix(np.outer(Y,Y)*kernel_matrix)
        #q对应于-αi=0
        q = cvxopt.matrix(np.ones(sample_num)*(-1))
        #不等式0<=αi<=C
        G_l = cvxopt.matrix(np.diag(np.ones(sample_num)*(-1)))
        S_l = cvxopt.matrix(np.zeros(sample_num))

        G_r = cvxopt.matrix(np.diag(np.ones(sample_num)))
        S_r = cvxopt.matrix(np.ones(sample_num)*self._c)

        G = cvxopt.matrix(np.vstack(G_l,G_r))
        S = cvxopt.matrix(np.vstack(S_l,S_r))
        #等式限制条件
        A = cvxopt.matrix(np.ones(sample_num))
        B = cvxopt.matrix(0.0)
        #实际地去计算出最优解
        solution = cvxopt.solvers.qp(p,q,G,S,A,B)
        return np.ravel(solution['x'])
    def _Generate_predictor(self,X,Y,optimized_alphas):
         '''
         :param X: 训练数据
         :param Y: 训练数据的标签
         :return:
         '''
         support_vector_indices = \
         optimized_alphas > _MIN_Support_Multiplier
         self.support_vectors = X[support_vector_indices]
         self.support_vectors_labels = Y[support_vector_indices]
         self.optimized_alphas = support_vector_indices[support_vector_indices]
         svm_predictor = SVM_predictor(self.support_vectors,self.support_vectors_labels,self.optimized_alphas,0.0,self._kernel)
         self.bias = np.mean([
            y-svm_predictor.predict(x)
            for (x,y) in zip(self.support_vectors,self.support_vectors_labels)
         ])
         return SVM_predictor(
             self.support_vectors,
             self.support_vectors_labels,
             self.optimized_alphas,
             self.bias,
             self._kernel
         )
    def predict(self,x):
        y = self.svmpredictor.predict(x)
    def fit(self,X,Y):
        self.optimized_alphas = self._fit_(X,Y)
        self.svmpredictor = self._Generate_predictor(X,Y,self.optimized_alphas)
        self.svmpredictor.log_info()

class SVM_predictor():
    def __init__(self,support_vectors,support_vectors_labels,optimized_alphas,bias,kernel):
        self._support_vectors = support_vectors
        self._support_vectors_labels = support_vectors_labels
        self._optimized_alphas = optimized_alphas
        self._bias=bias
        self._kernel = kernel
    def log_info(self):
        logging.info("Bias:%s",self._bias)
        logging.info("Weights: %s", self._optimized_alphas)
        logging.info("Support vectors: %s", self._support_vectors)
        logging.info("Support vector labels: %s", self._support_vectors_labels)
    def predict(self,x):
        '''
        :param x:待确定标签的数据
        :return: 输出的标签结果
        '''
        result = self._bias
        for x_i,y_i,a_i in zip(self._support_vectors,self._support_vectors_labels,self._optimized_alphas):
           result += y_i*a_i*self._kernel(x_i,x)
        return np.sign(result).item()
