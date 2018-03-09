from util import *

class DNCAccessor(object):
    """
    DNC Accessor, using autodiff, with batchsize=1
    """

    def __init__(self, R, N, W):

        self.R = R  # number of read heads
        self.N = N  # number of memory locations
        self.W = W  # number of columns in a memory location
         
        self.states = [self._init_state()]
        
    def _init_state(self):
        state = {}
        # linkage matrix
        state['L'] = np.zeros((self.N, self.N))
        # read_weighting
        state['rw'] = np.ones((self.R, self.N))*1e-6
        # precedence_weighting
        state['p'] = np.zeros((1, self.N))
        # write_weighting
        state['ww'] = np.ones((1, self.N))*1e-6
        # usage_vector
        state['u'] = np.ones((1, self.N))*1e-6
        return state
    
    def process_interface(self, interface):
        """
        Parse and process the input interface vector
        interface      - 1x(W*R+3W+5R+3) interface vector

        Return:
        rk_t         - RxW read keys
        rs_t         - Rx1 read strength
        wk_t         - 1xW write key
        ws_t         - 1x1 write strength
        e_t          - 1xW erase vector
        v_t          - 1xW write vector
        f_t          - Rx1 free gates
        ga_t         - 1x1 allocation gate
        gw_t         - 1x1 write gate
        pi_t         - Rx3 read modes
        """
        section_index = np.cumsum([self.R*self.W, self.R, self.W, 1, self.W, self.W, self.R, 1, 1, self.R*3])
        rk_t, rs_t, wk_t, ws_t, e_t, v_t, f_t, ga_t, gw_t, pi_t = np.split(interface, section_index[:-1], axis=1)
        rk_t = rk_t.reshape(self.R, self.W)
        rs_t = oneplus(rs_t.reshape(self.R, 1))
        wk_t = wk_t.reshape(1, self.W)
        ws_t = oneplus(ws_t)
        e_t = sigmoid(e_t.reshape(1, self.W))
        v_t = v_t.reshape(1, self.W)
        f_t = sigmoid(f_t.reshape(self.R, 1))
        ga_t = sigmoid(ga_t)
        gw_t = sigmoid(gw_t)
        pi_t = softmax(pi_t.reshape(self.R, 3))
        return rk_t, rs_t, wk_t, ws_t, e_t, v_t, f_t, ga_t, gw_t, pi_t  
        
    def content_weighting(self, mem, ks, betas):
        """
        The content based addressing method
        mem   - NxW memory     (N locations,  W entries)
        ks    - RxW lookup key (R heads, W entries)
        betas - 1xR lookup strength

        Return:
        RxN addressing matrix
        """
#         if len(ks.shape) < 2: ks = ks[np.newaxis, :] # deal with write head

        # Cosine Similarity
        n = np.dot(ks, mem.T)
        # einsum is fast: https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
        ks_inner_prod = np.einsum('ij,ij->i', ks, ks) + 1.e-20
        mem_inner_prod = np.einsum('ij,ij->i', mem, mem) + 1.e-20
        d = np.sqrt(np.einsum('i,j->ij', ks_inner_prod, mem_inner_prod)) # + 1.e-20 # prevent undefined cos similarity at 0 from breaking the code
        sim = betas * (n / d)
        
        return softmax(sim)
    
    def usage_vec(self, f_t, rw_prev, ww_prev, u_prev):
        """
        Update usage vector
        f_t           - Rx1 free gates
        rw_prev       - RxN previous read weightings (R read heads, N locations)
        ww_prev       - 1xN previous write weighting (R read heads, 1 location)
        u_prev        - 1xN previous usage vector

        Return:
        1xN new usage vector
        """
        # psi is the 1xN retention vector
        psi = np.ones_like(rw_prev) - f_t * rw_prev
        psi = np.prod(psi, axis=0)
        # u is the usage vector
        u = (u_prev + ww_prev - u_prev * ww_prev) * psi
        return u

    def allocation_weighting(self, u):
        """
        Dynamic memory allocation weighting mechanism
        u_prev        - 1xN current usage vector

        Return:
        1xN alloc_weighting
        """
        # phi is the indices list that would sort u in ascending order
        phi = np.argsort(u, axis=1).squeeze()
        inverse_perm = np.argsort(phi)
        
        # double check if this is differentiable
        sorted_alloc = (np.ones_like(u) - u[:,phi]) * shift_cumprod(u[:,phi])
        alloc_weighting = sorted_alloc[:,inverse_perm]
        return alloc_weighting

    def write_weighting(self, M_prev, wk_t, ws_t, u, gw_t, ga_t):
        """
        Write Weighting Mechanism
        
        M_prev        - NxW previous memory state
        wk_t          - 1xW write key
        ws_t          - 1x1 write strength
        f_t           - Rx1 free gates
        u             - 1xN current usage vector
        gw_t          - 1x1 write gate
        ga_t          - 1x1 allocation gate

        Return:
        1xN write_weighting 
        """
        c = self.content_weighting(M_prev, wk_t, ws_t)
        a = self.allocation_weighting(u)
        return gw_t * (ga_t * a + (1. - ga_t) * c)
        
    def temporal_memory_linkage(self, p_prev, ww, L_prev):
        """
        Temporal Linkage (TODO: Implement sparse link matrix)
        
        p_prev     - 1xN precedence weighting from last time step
        w_w        - 1xN write weighting
        L_prev     - NxN link matrix
        
        Return:
        1xN current precedence weighting
        NxN link matrix
        """
        # precedence weighting of the current timestep
        p_t = (1. - np.sum(ww)) * p_prev + ww
        L_t = (np.ones_like(L_prev) - (ww + ww.T)) * L_prev + ww.T * p_prev
        # Sanity check diag(L_t) should always be zero
        L_t = L_t - np.diag(np.diag(L_t))
        
        return p_t, L_t
    
    def read_weighting(self, M, rk_t, rs_t, rw_prev, L, pi_t):
        """
        Read Weighting
        
        M          - NxW memory matrix from current time step
        rk_t       - RxW read keys
        rs_t       - 1xR read strengths
        rw_prev    - RxN read weighting from previous time step
        L          - NxN link matrix from current time step
        pi_t       - Rx3 read modes
        
        Return:
        RxN Read Weighting
        """
        # content weighting
        c = self.content_weighting(M, rk_t, rs_t)
        # forward weighting
        f_t = np.dot(rw_prev, L)
        # backward weighting
        b_t = np.dot(rw_prev, L.T)
        # interpolates using read modes
        read_weighting = pi_t[:,0,np.newaxis] * b_t + pi_t[:,1,np.newaxis] * c + pi_t[:,2,np.newaxis] * f_t
        return read_weighting
    
    def read(self, M, rw):
        """
        Read from memory
        
        M          - NxW memory matrix from current time step
        rw         - RxN read weighting
        
        Return:
        RxW Stacked Read Vectors
        """
        return np.dot(rw, M)
    
    def write(self, M, e_t, v_t, ww):
        """
        Write to memory
        
        M          - NxW memory matrix from current time step
        e_t        - 1xW erase vector
        v_t        - 1xW write vector
        ww         - 1xN write weighting
        
        Return:
        NxW updated memory
        """
        return (np.ones_like(M) - np.dot(ww.T, e_t)) * M + np.dot(ww.T, v_t)
    
    def step_forward(self, M_prev, interface):
        """
        Forward inference given a inference vector and previous memory state
        M_prev         - NxW previous memory state
        interface      - 1x(W*R+3W+5R+3) interface vector
        
        Return:
        RxW Stacked Read Vectors
        """
        rk_t, rs_t, wk_t, ws_t, e_t, v_t, f_t, ga_t, gw_t, pi_t = self.process_interface(interface)
        _s = self.states[-1] # previous state
        L_prev, rw_prev, p_prev, ww_prev, u_prev = _s['L'], _s['rw'], _s['p'], _s['ww'], _s['u']
        
        u = self.usage_vec(f_t, rw_prev, ww_prev, u_prev)
#         print "write: "
        ww = self.write_weighting(M_prev, wk_t, ws_t, u, gw_t, ga_t)
        M = self.write(M_prev, e_t, v_t, ww)
        p, L = self.temporal_memory_linkage(p_prev, ww, L_prev)
#         print "read: "
        rw = self.read_weighting(M, rk_t, rs_t, rw_prev, L, pi_t)
        
        self.states.append(dict(zip(['u', 'ww', 'p', 'L', 'rw'],[u, ww, p, L, rw])))
        
        read_vec = self.read(M, rw)
        return M, read_vec
    
    
    def step_forward_breakage_test(self, M_prev, interface):
        """
        Test step forward gradient breakage point, uncomment each of the return clause return intermediate results
        """
        rk_t, rs_t, wk_t, ws_t, e_t, v_t, f_t, ga_t, gw_t, pi_t = self.process_interface(interface)
#         return rk_t, rs_t, wk_t, ws_t, e_t, v_t, f_t, ga_t, gw_t, pi_t

        _s = self.states[-1] # previous state
        L_prev, rw_prev, p_prev, ww_prev, u_prev = _s['L'], _s['rw'], _s['p'], _s['ww'], _s['u']

        u = self.usage_vec(f_t, rw_prev, ww_prev, u_prev)
#         return u

        ww = self.write_weighting(M_prev, wk_t, ws_t, u, gw_t, ga_t)
#         return ww
        ww_prime = np.copy(ww) # stop gradient from flowing into M
        M = self.write(M_prev, e_t, v_t, ww)
#         return M

        p, L = self.temporal_memory_linkage(p_prev, ww, L_prev)
#         return p, L

        rw = self.read_weighting(M, rk_t, rs_t, rw_prev, L, pi_t)
#         return rw

        self.states.append(dict(zip(['u', 'ww', 'p', 'L', 'rw'],[u, ww, p, L, rw])))

        read_vec = self.read(M, rw)
        return read_vec

    #     return M, read_vec

# Testing
# accessor = DNCAccessor(2,3,4) #R, N, W
# interface = nprn(1,2*4+3*4+5*2+3)
# memory = nprn(3,4)
# mem,vec = accessor.step_forward(memory, interface)
# print mem
# print vec
# print len(accessor.states)
# print accessor.states
