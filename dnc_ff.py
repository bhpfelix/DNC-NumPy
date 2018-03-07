from util import *
from accessor import DNCAccessor

class DNCFF(object):
    """
    DNC with feedforward controller, using autodiff, with batchsize=1
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
        
        # two layer feedforward weights
        p['W_1'] = nprn(self.input_size + self.R * self.W, self.hidden_size)*0.1
        p['b_1'] = np.zeros((1, self.hidden_size))
        p['W_2'] = nprn(self.hidden_size, self.output_size)*0.1
        p['b_2'] = np.zeros((1, self.output_size))
        
        p['W_y'] = nprn(self.output_size, self.output_size)*0.1
        p['W_xi'] = nprn(self.output_size, self.R*self.W + 3*self.W + 5*self.R + 3)*0.1
        
        # output weights
        p['W_r'] = nprn(self.R*self.W, self.output_size)*0.1
        return p
    
    def _init_state(self):
        state = {}
        # not initializing to zero can prevent nan in loss
        # memory matrix
        state['M'] = np.zeros((self.N, self.W)) 
        # read vector
        state['rv'] = np.ones((self.R, self.W))*1e-6
        self.states = [state]
    
    def nn_step_forward(self, params, x_t, rv_prev):
        """
        nn step forward

        x_t        - 1xX input
        rv_prev    - RxW read vector from prev step
        params     - lstm parameter dictionary, included here for the purpose of backprop

        Return:
        v_t        - 1xY output vector
        interface  - 1x(W*R+3W+5R+3) interface vector
        """
        X_t = np.concatenate((x_t, rv_prev.reshape(1,-1)), axis=1)
        h_1 = np.dot(X_t, params['W_1']) + params['b_1']
        h_1 = np.tanh(h_1)
        o_1 = np.dot(h_1, params['W_2']) + params['b_2']
        o_1 = np.tanh(o_1)

        v_t = np.dot(o_1, params['W_y'])
        interface = np.dot(o_1, params['W_xi'])
        return v_t, interface
    
    def step_forward(self, params, x_t):
        if self.states is None: self._init_state()
        
        _s = self.states[-1]
        M_prev, rv_prev = _s['M'], _s['rv']
        
        v_t, interface = self.nn_step_forward(params, x_t, rv_prev)
#         print "interface!!!: ", interface
        M_t, rv_t = self.accessor.step_forward(M_prev, interface)
        state = dict(zip(['M', 'rv'], [M_t, rv_t]))
        self.states.append(state)
        
        out = v_t + np.dot(rv_t.reshape(1,-1), params['W_r'])
        return out
         
# # Test
# dnc = DNCFF(input_size=10, output_size=10, hidden_size=32, R=1, N=10, W=1)
# dnc_params = dnc._init_params()
# inpt = np.array([[0., 1., 0., 0., 1., 0., 0., 1., 0., 1.]])
# print inpt
# print dnc.step_forward(dnc_params, inpt)
