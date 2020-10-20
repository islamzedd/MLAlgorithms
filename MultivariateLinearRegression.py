features=[[1,2,3],
         [1,4,9],
         [1,5,0]]
prices=[5,
        13,
        5]
theta = [0]*len(features[0])

LEARNING_RATE=0.01
NO_TRAINING_EXAMPLES = len(features)
EPSILON=0.000000001


#prediction function to find the price given the features and thetas
def predict(features,theta):
    y=0
    for i in range(len(features)):
       y+= theta[i]*features[i]
    return y

def cost(featurs,prices,theta):
    sum=0
    for i in range(NO_TRAINING_EXAMPLES):
        sum += (predict(features[i], theta) - prices[i]) ** 2
    cost = sum / (2 * NO_TRAINING_EXAMPLES)
    return cost

def gradient_descent(features,prices,theta):
    old_cost = cost(features, prices, theta)
    while True:
        temp=theta
        for i in range(len(theta)):
            sum=0
            for j in range(NO_TRAINING_EXAMPLES):
                sum += (predict(features[j],theta)-prices[j])*features[j][i]
            temp[i]=theta[i]-(LEARNING_RATE/NO_TRAINING_EXAMPLES)*sum
        theta=temp

        new_cost = cost(features, prices, theta)

        # test for convergence
        if abs(old_cost - new_cost) < EPSILON:
            break
        else:
            old_cost = new_cost
    return theta

print(gradient_descent(features,prices,theta))