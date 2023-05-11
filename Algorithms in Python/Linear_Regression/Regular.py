#REGULAR LINEAR REGRESSION
import numpy as np
class LinearRegression:
    def __init__(self,lr=.01,iterations=10000,bias=0):
        self.lr=lr
        self.iterations=iterations
        self.weights=None
        self.bias=bias 
        
    def fit(self,x,y):
        samples,features=x.shape
        self.weights=np.zeros(features)
        
        for i in range (0,self.iterations):
            prediction= y-((np.dot(x,self.weights)+ self.bias))
            gradient=(-2/samples) *(np.dot(np.transpose(x), prediction))
            self.weights=self.weights-(self.lr*gradient)
            
            self.bias=self.bias-(self.lr*((-2/samples)*np.sum(y-(np.dot(x,self.weights)))))
    
    def predict(self,z):
        return (np.dot(z,self.weights)+ self.bias)
