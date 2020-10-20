import numpy as np
import math

#starting parameters
theta=np.array([
    [0.00001],
    [0.00001]
])

#training set features
X =np.array([
[1,0],
[1,1],
[1,2],
[1,3],
[1,4],
[1,5],
[1,6],
[1,7],
[1,8],
[1,9],
[1,10],
[1,11],
[1,12],
[1,13],
[1,14],
[1,15],
[1,16],
[1,17],
[1,18],
[1,19],
[1,20],
[1,21],
[1,22],
[1,23],
[1,24],
[1,25],
[1,26],
[1,27],
[1,28],
[1,29],
[1,30],
[1,31],
[1,32],
[1,33],
[1,34],
[1,35],
[1,36],
[1,37],
[1,38],
[1,39],
[1,40],
[1,41],
[1,42],
[1,43],
[1,44],
[1,45],
[1,46],
[1,47],
[1,48],
[1,49]
])

#training set classes
y=  np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1]
])

LEARNING_RATE=0.01
NO_TRAINING_EXAMPLES = len(X)
EPSILON=0.000000001


#predict the probabilty of being in class (1)
def predict(theta,features):
    z=np.dot(np.transpose(theta),features)
    sigmoid = 1/(1+math.exp(-z))
    return sigmoid

#cost function to direct and halt gradient descent
def cost(theta,features,classification):
    sum = 0
    for i in range(NO_TRAINING_EXAMPLES):
        prediction=predict(theta,features[i])
        if classification[i]==1:
            sum += -math.log10(prediction)
        else:
            sum += -math.log10(1-prediction)
    cost = (1/NO_TRAINING_EXAMPLES)*sum
    return cost

#use gradient descent to get the best parameters with the lowest cost for the given set
def gradient_descent(theta,features,classification):
    old_cost=cost(theta,features,classification)
    while True:
        #evaluate the partial derivative for theta0 and theta1
        sum0 = 0
        sum1 = 0
        for i in range(NO_TRAINING_EXAMPLES):
            prediction=predict(theta,features[i])
            sum0 += (prediction-classification[i]) * features[i][0]
            sum1 += (prediction - classification[i]) * features[i][1]

        #update both thetas simultaneously
        #print(sum0)
        #print(sum1)
        theta[0] = theta[0] - (LEARNING_RATE / NO_TRAINING_EXAMPLES) * sum0
        theta[1] = theta[1] - (LEARNING_RATE / NO_TRAINING_EXAMPLES) * sum1
        #print(theta[0])
        #print(theta[1])

        new_cost=cost(theta,features,classification)

        #test for convergence
        if abs(old_cost-new_cost) < EPSILON:
            break
        else:
            old_cost=new_cost

    return theta

newtheta=gradient_descent(theta,X,y)
print(newtheta)
print(predict(newtheta,[1,24]))
print(cost(newtheta,X,y))

