#REGULAR LINEAR REGRESSION
import numpy as np
class LinearRegression:
    def __init__(self):
        self.lr=0.01
        self.iterations=10000
        self.weights=None
        self.bias=0 
        
    def fit(self,x,y):
        samples,features=x.shape
        self.weights=np.zeros(features)
        
        for i in range (0,self.iterations):
            gradient=np.dot(np.transpose(x),(y-(np.dot(x,self.weights)+ self.bias)))
            self.weights=self.weights-(self.lr*gradient)
            
            self.bias=self.bias-(self.lr*np.sum(y-(np.dot(x,self.weights))))
    
    def predict(self,z):
        return np.dot(z,self.weights)+ self.bias
