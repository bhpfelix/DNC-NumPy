from util import *
from accessor import DNCAccessor

class DNC(object):
    """
    DNC with LSTM controller, using autodiff, with batchsize=1
    """

    def __init__(self, input_size, output_size, hidden_size, R, N, W):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.R = R
        self.N = N  # the number of memory locations
        self.W = W # the number of columns in a memory location
        
        self.accessor = DNCAccessor(R, N, W)
        self.states = None
     
    def _init_params(self):
        p = {} # maps parameter names to tensors
        
        # single layer LSTMController weights
        p['W_i'] = nprn(self.input_size + self.R * self.W + self.hidden_size, self.hidden_size)*0.1
        p['b_i'] = np.zeros((1, self.hidden_size))
        p['W_f'] = nprn(self.input_size + self.R * self.W + self.hidden_size, self.hidden_size)*0.1
        p['b_f'] = np.zeros((1, self.hidden_size))
        p['W_s'] = nprn(self.input_size + self.R * self.W + self.hidden_size, self.hidden_size)*0.1
        p['b_s'] = np.zeros((1, self.hidden_size))
        p['W_o'] = nprn(self.input_size + self.R * self.W + self.hidden_size, self.hidden_size)*0.1
        p['b_o'] = np.zeros((1, self.hidden_size))
        
        p['W_y'] = nprn(self.hidden_size, self.output_size)*0.1
        p['W_xi'] = nprn(self.hidden_size, self.R*self.W + 3*self.W + 5*self.R + 3)*0.1
        
        # output weights
        p['W_r'] = nprn(self.R*self.W, self.output_size)*0.1
        return p

    def _init_state(self):
        state = {}
        # not initializing to zero can prevent nan in loss
        # memory matrix
        state['M'] = np.zeros((self.N, self.W))
        # lstm hidden state
        state['h'] = np.zeros((1, self.hidden_size))
        # lstm cell state
        state['s'] = np.zeros((1, self.hidden_size))
        # read vector
        state['rv'] = np.ones((self.R, self.W))*1e-6
        self.states = [state]
    
    def lstm_step_forward(self, params, x_t, rv_prev, h_prev, s_prev):
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
        X_t = np.concatenate((x_t, rv_prev.reshape(1,-1), h_prev), axis=1)
        i_t = sigmoid(np.dot(X_t, params['W_i']) + params['b_i'])
        f_t = sigmoid(np.dot(X_t, params['W_f']) + params['b_f'])
        s_t = f_t * s_prev + i_t * np.tanh(np.dot(X_t, params['W_s']) + params['b_s'])
        o_t = sigmoid(np.dot(X_t, params['W_o']) + params['b_o'])
        h_t = o_t * np.tanh(s_t)

        v_t = np.dot(h_t, params['W_y'])
        interface = np.dot(h_t, params['W_xi'])
        return h_t, s_t, v_t, interface
    
    def step_forward(self, params, x_t):
        if self.states is None: self._init_state()
        
        _s = self.states[-1]
        M_prev, h_prev, s_prev, rv_prev = _s['M'], _s['h'], _s['s'], _s['rv']
        
        h_t, s_t, v_t, interface = self.lstm_step_forward(params, x_t, rv_prev, h_prev, s_prev)
#         print "interface!!!: ", interface
        M_t, rv_t = self.accessor.step_forward(M_prev, interface)
        state = dict(zip(['M', 'h', 's', 'rv'], [M_t, h_t, s_t, rv_t]))
        self.states.append(state)
        
        out = v_t + np.dot(rv_t.reshape(1,-1), params['W_r'])
        return out
        
# # Test
# dnc = DNC(input_size=10, output_size=10, hidden_size=1, R=1, N=10, W=1)
# dnc_params = dnc._init_params()
# inpt = np.array([[0., 1., 0., 0., 1., 0., 0., 1., 0., 1.]])
# print inpt
# print dnc.step_forward(dnc_params, inpt)
