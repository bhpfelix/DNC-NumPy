from util import *

class LSTM(object):
    """
    Vanilla LSTM, using autodiff, with batchsize=1
    """

    def __init__(self, input_size, output_size, hidden_size):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

    def _init_params(self):
        p = {} # maps parameter names to tensors

        # single layer LSTMController weights
        p['W_i'] = nprn(self.input_size + self.hidden_size, self.hidden_size)*0.1
        p['b_i'] = np.zeros((1, self.hidden_size))
        p['W_f'] = nprn(self.input_size + self.hidden_size, self.hidden_size)*0.1
        p['b_f'] = np.zeros((1, self.hidden_size))
        p['W_s'] = nprn(self.input_size + self.hidden_size, self.hidden_size)*0.1
        p['b_s'] = np.zeros((1, self.hidden_size))
        p['W_o'] = nprn(self.input_size + self.hidden_size, self.hidden_size)*0.1
        p['b_o'] = np.zeros((1, self.hidden_size))

        p['W_h2c'] = nprn(self.hidden_size, 1)*0.1
        p['b_h2c'] = np.zeros((1, 1))
        return p

    def _init_state(self):
        return np.zeros((1, self.hidden_size)), np.zeros((1, self.hidden_size))

    def lstm_step_forward(self, params, x_t, h_prev, s_prev):
        """
        LSTM step forward

        x_t        - 1xX input
        rv_prev    - RxW read vector from prev step
        h_prev     - 1xH previous hidden state
        s_prev     - 1xH previous cell state
        params     - lstm parameter dictionary, included here for the purpose of backprop

        Return:
        h_t        - 1xH current hidden state
        s_t        - 1xH current cell state
        v_t        - 1xY output vector
        interface  - 1x(W*R+3W+5R+3) interface vector
        """
        if h_prev is None or s_prev is None:
            h_prev, s_prev = self._init_state()
        X_t = np.concatenate((x_t, h_prev), axis=1)
        i_t = sigmoid(np.dot(X_t, params['W_i']) + params['b_i'])
        f_t = sigmoid(np.dot(X_t, params['W_f']) + params['b_f'])
        s_t = f_t * s_prev + i_t * np.tanh(np.dot(X_t, params['W_s']) + params['b_s'])
        o_t = sigmoid(np.dot(X_t, params['W_o']) + params['b_o'])
        h_t = o_t * np.tanh(s_t) # output gate suppressed result

        out = sigmoid(np.dot(h_t, params['W_h2c']) + params['b_h2c'])
        return out, h_t, s_t


from util import *
from lstm import LSTM
from dnc_lstm import DNC
from dnc_ff import DNCFF
from autograd import grad
from autograd.misc.optimizers import rmsprop, adam

# create a sequence classification instance
def get_sequence(n_timesteps):
    # create a sequence of random numbers in [0,1]
    X = np.array([np.random.rand() for _ in range(n_timesteps)])
    # calculate cut-off value to change class values
    limit = n_timesteps/4.0
    # determine the class outcome for each item in cumulative sequence
    y = np.array([[0 if x < limit else 1 for x in np.cumsum(X)]])
    return X, y



# Toy LSTM example on sequence classification.
"""
Toy problem from: https://machinelearningmastery.com/sequence-prediction-problems-learning-lstm-recurrent-neural-networks/

A binary label (0 or 1) is associated with each input. The output values are all 0. Once the cumulative sum of the input values in the sequence exceeds a threshold, then the output value flips from 0 to 1.

A threshold of 1/4 the sequence length is used.

For example, below is a sequence of 10 input timesteps (X):

0.63144003 0.29414551 0.91587952 0.95189228 0.32195638 0.60742236 0.83895793 0.18023048 0.84762691 0.29165514

The corresponding classification output (y) would be:

0 0 0 1 1 1 1 1 1 1
"""

lstm = LSTM(1,1,32)
lstm_params = lstm._init_params()

def loss_fn(pred, target):
    one = np.ones_like(pred)
    epsilon = 1.e-20 # to prevent log(0)
    a = target * np.log(pred + epsilon)
    b = (one - target) * np.log(one - pred + epsilon)
    return np.mean(- (a + b))

def print_training_prediction(params, iters):

    X, y = get_sequence(10)
    h, s = None, None
    result = []
    for x in X:
        out, h, s = lstm.lstm_step_forward(params, np.array([[x]]), h, s)
        result.append(out)
    result = np.hstack(result)
    print y
    print np.around(result).astype('int')

def training_loss(params, iters):
    X, y = get_sequence(10)
    h, s = None, None
    result = []
    for x in X:
        out, h, s = lstm.lstm_step_forward(params, np.array([[x]]), h, s)
        result.append(out)
    result = np.hstack(result)
    loss = loss_fn(result, y)

    # regularization
#     reg = l2(params['W_1']) + l2(params['W_2']) + l2(params['b_1']) + l2(params['b_2'])
    return loss

def callback(weights, iters, gradient):
    if iters % 100 == 0:
        print("Iteration", iters, "Train loss:", training_loss(weights, 0))
        print_training_prediction(weights, iters)

# Build gradient of loss function using autograd.
training_loss_grad = grad(training_loss)

print("Training LSTM...")
# trained_params = adam(training_loss_grad, dnc_params, step_size=0.001,
#                       num_iters=1000000, callback=callback)
trained_params = rmsprop(training_loss_grad, lstm_params, step_size=0.001,
                      num_iters=1000000, callback=callback)