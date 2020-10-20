import numpy as np

features=[[1,2,3],
          [1,4,9],
          [1,5,0]]
X = np.array(features)

prices = [5,
          13,
          5]
y=np.array(prices)

theta = [0]*len(features[0])
theta = np.transpose(np.array(theta))

LEARNING_RATE=0.01
NO_TRAINING_EXAMPLES = len(features)
EPSILON=0.000000001

def predict(X,theta):
    y=np.matmul(np.transpose(theta),X)
    return y

def cost(X,y,theta):
    return (1/(2*NO_TRAINING_EXAMPLES))*np.transpose((X@theta - y))@(X@theta - y)

def gradient_descent(X,y,theta):
    old_cost = cost(X, y, theta)
    while True:
        sum=0
        for i in range(NO_TRAINING_EXAMPLES):
            sum += (predict(X[i],theta)-y[i])*X[i]
        delta = sum/NO_TRAINING_EXAMPLES
        theta=theta-LEARNING_RATE*np.transpose(delta)

        new_cost=cost(X,y,theta)

        # test for convergence
        if abs(old_cost - new_cost) < EPSILON:
            break
        else:
            old_cost = new_cost
    return theta
print(gradient_descent(X,y,theta))