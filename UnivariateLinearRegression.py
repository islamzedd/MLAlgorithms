features = [1,2,3]
prices = [1,2,3]
theta = [0,0]

LEARNING_RATE=0.01
NO_TRAINING_EXAMPLES=len(features)
EPSILON=0.000000001

#Cost function to calculate half the average of the squared errors for the given theta
def cost(features, prices, theta):
    sum = 0
    for i in range(NO_TRAINING_EXAMPLES):
        sum += (predict(features[i],theta)-prices[i])**2
    cost = sum/(2*NO_TRAINING_EXAMPLES)
    return cost

#prediction function to find the price given the feature and theta
def predict(feature,theta):
    y=theta[0]+theta[1]*feature
    return y

#gradient descent algorithm to find the value of theta that makes the prediction most accurate
#i.e causing the cost function to be minimum
def gradient_descent(features,prices,theta):
    old_cost=cost(features,prices,theta)
    while True:

        #evaluate the partial derivative for theta0 and theta1
        sum0 = 0
        sum1 = 0
        for i in range(NO_TRAINING_EXAMPLES):
            sum0 += (predict(features[i],theta)-prices[i])
            sum1 += (predict(features[i], theta) - prices[i]) * features[i]

        #update both thetas simultaneously
        theta[0] = theta[0] - (LEARNING_RATE / NO_TRAINING_EXAMPLES) * sum0
        theta[1] = theta[1] - (LEARNING_RATE / NO_TRAINING_EXAMPLES) * sum1

        new_cost=cost(features,prices,theta)

        #test for convergence
        if abs(old_cost-new_cost) < EPSILON:
            break
        else:
            old_cost=new_cost

    return theta


print(gradient_descent(features,prices,theta))