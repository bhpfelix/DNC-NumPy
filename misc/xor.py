from util import *

# Test simple ff network
xor_params = {
    'W_1':nprn(2,4),
    'W_2':nprn(4,1),
    'b_1':np.zeros((1,4)),
    'b_2':np.zeros((1,1)),
}

dataset = [[0,0],[0,1],[1,0],[1,1]]
dataset = np.array(dataset)
targets = np.array([0,1,1,0])

def feedforward(params, x_t):
    h_1 = np.dot(x_t, params['W_1']) + params['b_1']
    h_1 = sigmoid(h_1)
    o_1 = np.dot(h_1, params['W_2']) + params['b_2']
    o_1 = sigmoid(o_1)
    return o_1

def loss_fn(out, target):
    return np.sqrt(np.square(out-target))

def print_training_prediction(params):
    for row in dataset:
        out = feedforward(params, row[np.newaxis, :])
        print row, out

def training_loss(params, iter):
    loss = 0
    for idx, row in enumerate(dataset):
        out = feedforward(params, row[np.newaxis, :])
        loss = loss + loss_fn(out, targets[idx])
    return loss

def callback(weights, iter, gradient):
    if iter % 500 == 0:
        print("Iteration", iter, "Train loss:", training_loss(weights, 0))
        print_training_prediction(weights)

# Build gradient of loss function using autograd.
training_loss_grad = grad(training_loss)

print("Training XOR...")
# trained_params = adam(training_loss_grad, dnc_params, step_size=0.001,
#                       num_iters=1000, callback=callback)
trained_params = rmsprop(training_loss_grad, xor_params, step_size=0.001,
                      num_iters=10000, callback=callback)