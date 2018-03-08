import unittest
from util import *
from accessor import DNCAccessor
from dnc_lstm import DNC
from dnc_ff import DNCFF

R, N, W = 2,3,4

## Forward Test Cases:

class BaseAccessorTest(unittest.TestCase):
    def setUp(self):
        self.R = R
        self.N = N
        self.W = W
        self.accessor = DNCAccessor(R, N, W) #R, N, W
        self.p = {
            'interface':nprn(1,R*W+3*W+5*R+3)
        }
        
class InterfaceParsing(BaseAccessorTest):
    def runTest(self):
        rk_t, rs_t, wk_t, ws_t, e_t, v_t, f_t, ga_t, gw_t, pi_t = self.accessor.process_interface(self.p['interface'])
        self.assertTrue(np.all(rs_t >= 1.))
        self.assertTrue(np.all(ws_t >= 1.))
        self.assertTrue(np.all(e_t >= 0.) and np.all(e_t <= 1.))
        self.assertTrue(np.all(f_t >= 0.) and np.all(f_t <= 1.))
        self.assertTrue(np.all(ga_t >= 0.) and np.all(ga_t <= 1.))
        self.assertTrue(np.all(gw_t >= 0.) and np.all(gw_t <= 1.))
        self.assertTrue(np.all(pi_t >= 0.) and np.all(pi_t <= 1.))
        
class ContentAddressing(BaseAccessorTest):
    def runTest(self):
        rk_t, rs_t, wk_t, ws_t, e_t, v_t, f_t, ga_t, gw_t, pi_t = self.accessor.process_interface(self.p['interface'])
        mem = nprn(3,4)
        normed_mem = mem / np.expand_dims(np.linalg.norm(mem, axis=1), 1)
        normed_key = rk_t / np.expand_dims(np.linalg.norm(rk_t, axis=1), 1)
        normed_sim = np.dot(normed_key, normed_mem.T) * rs_t
        normed_sim = softmax(normed_sim)
        self.assertTrue(np.allclose(normed_sim, self.accessor.content_weighting(mem, rk_t, rs_t)))

class UsageVecUpdate(BaseAccessorTest):
    def runTest(self):
        rk_t, rs_t, wk_t, ws_t, e_t, v_t, f_t, ga_t, gw_t, pi_t = self.accessor.process_interface(self.p['interface'])
        f_t = np.array([[0.25],[0.5]])
        rw_prev = np.ones((2,3))*0.5
        u_prev = np.array([0.5,0.75,0.9])
        ww_prev = np.array([0.1,0.1,0.1])
        result = 0.65625*np.ones((1,3)) *  np.array([0.55,0.775,0.91])
        self.assertTrue(np.allclose(result, self.accessor.usage_vec(f_t, rw_prev, ww_prev, u_prev)))
        
class AllocWeight(BaseAccessorTest):
    def runTest(self):
        u = np.array([[0.2, 0.1, 0.5]])
        self.assertTrue(np.allclose(np.array([[0.08,0.9,0.01]]), self.accessor.allocation_weighting(u)))   
             
class WriteWeight(BaseAccessorTest):
    def runTest(self):
        rk_t, rs_t, wk_t, ws_t, e_t, v_t, f_t, ga_t, gw_t, pi_t = self.accessor.process_interface(self.p['interface'])
        mem = nprn(3,4)
        normed_mem = mem / np.expand_dims(np.linalg.norm(mem, axis=1), 1)
        normed_key = wk_t / np.expand_dims(np.linalg.norm(wk_t, axis=1), 1)
        normed_sim = np.dot(normed_key, normed_mem.T) * ws_t
        c = softmax(normed_sim)
        
        u = np.array([[0.2, 0.1, 0.5]])
        a = np.array([[0.08,0.9,0.01]])
        
        ww_test = gw_t * (ga_t * a + (1. - ga_t) * c)
        self.assertTrue(np.allclose(ww_test, self.accessor.write_weighting(mem, wk_t, ws_t, u, gw_t, ga_t)))      
        
class LinkMat(BaseAccessorTest):
    def runTest(self):
        p_prev = np.array([[0.2,0.1,0.4]])
        ww = np.array([[0.1,0.2,0.3]])
        L_prev = np.array([[0,0.2,0.4],[0.3,0.,0.6],[0.5,0.4,0.]])
        _p_t = np.array([0.18,0.24,0.46])
        _L = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    _L[i][j] = (1. - ww[0,i] - ww[0,j]) * L_prev[i][j] + ww[0,i] * p_prev[0,j]
        p_t, L = self.accessor.temporal_memory_linkage(p_prev, ww, L_prev)
        self.assertTrue(np.allclose(_p_t, p_t))  
        self.assertTrue(np.allclose(_L, L))

        
class LinkMat2(BaseAccessorTest):
    def runTest(self):
        L = np.zeros((self.N, self.N))
        p = np.zeros((1, self.N))
        
        for i in range(5):
            ww = np.random.rand(1,self.N)
            ww /= np.sum(ww) + 1.
            
            if i == 3:
                ww = np.array([[1,0,0]])
            if i == 4:
                ww = np.array([[0,1,0]])
                
            p, L =  self.accessor.temporal_memory_linkage(p, ww, L)
            
        self.assertTrue(np.all(L >= 0) and np.all(L <= 1))
        self.assertTrue(np.sum(np.diag(L))==0)
        self.assertTrue(np.all(np.sum(L, axis=0) <= 1.) and np.all(np.sum(L, axis=1) <= 1.))
        self.assertTrue(np.all(np.array([1,0,0]) == L[1]))
        
        # Test forward, backward
        rw_prev = np.array([[0,1,0],[1,0,0]])
        self.assertTrue(np.all(np.dot(rw_prev, L)[0] == np.array([1,0,0])))
        self.assertTrue(np.all(np.dot(rw_prev, L.T)[1] == np.array([0,1,0])))
        
        
## Gradient Test Cases
def auto_diff(func, param):
    """
    Used to wrap the arguments to function for gradient testing purpose
    func: target function to check gradient
    param: kwargs to func
    """
    def foo(param):
        out = func(**param)
        res = 0
        for item in out:
            res = res + np.sum(item)
        return res

    grad_foo = grad(foo)
    return grad_foo(param)

def numeric_diff(func, param, delta=1e-6):
    """
    Used to wrap the arguments to function for gradient testing purpose
    func: target function to check gradient
    param: kwargs to func
    """
    def foo(param):
        out = func(**param)
        res = 0
        for item in out:
            res = res + np.sum(item)
        return res

    results = {}
    for k, v in param.items():
        shape = v.shape
        grad_v = np.zeros_like(v).flatten()
        for i in range(v.size):
            temp1 = param.copy()
            temp2 = param.copy()
            temp_val1 = v.flatten()
            temp_val2 = v.flatten()
            temp_val1[i] -= delta
            temp_val2[i] += delta
            temp1[k] = temp_val1.reshape(shape)
            temp2[k] = temp_val2.reshape(shape)
            val1 = foo(temp1)
            val2 = foo(temp2)
            grad_val = (val2 - val1) / (2.*delta)
            grad_v[i] = grad_val
        results[k] = grad_v.reshape(shape)
    return results

def get_grad(func_name, param):
    """
    func_name  -    string of function name
    """
    accessor1 = DNCAccessor(R,N,W)
    accessor2 = DNCAccessor(R,N,W)
    numdiff = numeric_diff(getattr(accessor1, func_name), param, 1e-6)
    autodiff = auto_diff(getattr(accessor2, func_name), param)
    return numdiff, autodiff
    

class NumericalDiffTest(unittest.TestCase):
    def runTest(self):
        def test_func(a,b,c):
            return (a**2) / 2. + (b**3) / 3. + (c**4) / 4.
        a = nprn(2,3)
        b = nprn(2,3)
        c = nprn(2,3)
        param = {'a':a, 'b':b, 'c':c}
        numdiff = numeric_diff(test_func, param, 1e-6)
        autodiff = auto_diff(test_func, param)
        for k in param.keys():
            self.assertTrue(np.allclose(numdiff[k], autodiff[k]))
    
class ContentAddressingDividedByZero(BaseAccessorTest):
    def runTest(self):
        mem = np.vstack([np.ones((1,self.W)), np.zeros((2,self.W))])
        param = {'mem':mem, 'ks':nprn(1,4), 'betas':oneplus(nprn(1,1))}
        grad = auto_diff(self.accessor.content_weighting, param)
        for k, v in grad.items():
            self.assertTrue(np.all(np.isfinite(v)))
            
class InterfaceGradient(unittest.TestCase):
    def runTest(self):
        param = {'interface':nprn(1,2*4+3*4+5*2+3)}
        numdiff, autodiff = get_grad('process_interface', param)
        for k in param.keys():
            self.assertTrue(np.allclose(numdiff[k], autodiff[k]))
            
class ContentWeightGradient(unittest.TestCase):
    def runTest(self):
        param = {'mem':nprn(3,4), 'ks':nprn(2,4), 'betas':oneplus(nprn(1,1))}
        numdiff, autodiff = get_grad('content_weighting', param)
        for k in param.keys():
            self.assertTrue(np.allclose(numdiff[k], autodiff[k]))

class UsageGradient(unittest.TestCase):
    def runTest(self):
        param = {'f_t':nprn(2,1), 'rw_prev':nprn(2,3), 'ww_prev':nprn(1,3), 'u_prev':nprn(1,3)}
        numdiff, autodiff = get_grad('usage_vec', param)
        for k in param.keys():
            self.assertTrue(np.allclose(numdiff[k], autodiff[k]))

class AllocationGradient(unittest.TestCase):
    def runTest(self):
        param = {'u':nprn(1,3)}
        numdiff, autodiff = get_grad('allocation_weighting', param)
        for k in param.keys():
            self.assertTrue(np.allclose(numdiff[k], autodiff[k]))  
            
class WriteWeightingGradient(unittest.TestCase):
    def runTest(self):
        param = {'M_prev':nprn(3,4), 'wk_t':nprn(1,4), 'ws_t':nprn(1,1), 'u':nprn(1,3), 'gw_t':nprn(1,1), 'ga_t':nprn(1,1)}
        numdiff, autodiff = get_grad('write_weighting', param)
        for k in param.keys():
            self.assertTrue(np.allclose(numdiff[k], autodiff[k]))  
            
class LinkageGradient(unittest.TestCase):
    def runTest(self):
        param = {'p_prev':nprn(1,3), 'ww':nprn(1,3), 'L_prev':nprn(3,3)}
        numdiff, autodiff = get_grad('temporal_memory_linkage', param)
        for k in param.keys():
            self.assertTrue(np.allclose(numdiff[k], autodiff[k]))
        
class ReadWeightingGradient(unittest.TestCase):
    def runTest(self):
        param = {'M':nprn(3,4), 'rk_t':nprn(2,4), 'rs_t':nprn(2,1), 'rw_prev':nprn(2,3), 'L':nprn(3,3), 'pi_t':nprn(2,3)}
        numdiff, autodiff = get_grad('read_weighting', param)
        for k in param.keys():
            self.assertTrue(np.allclose(numdiff[k], autodiff[k]))
            
class StepForwardGradient(unittest.TestCase):
    def runTest(self):
        param = {'M_prev':nprn(3,4), 'interface':nprn(1,2*4+3*4+5*2+3)}
        numdiff, autodiff = get_grad('step_forward', param)
        for k in param.keys():
            print k
            print numdiff[k]
            print autodiff[k]
        for k in param.keys():
            self.assertTrue(np.allclose(numdiff[k], autodiff[k]))
           
        
if __name__ == '__main__':
    unittest.main()
    
    
