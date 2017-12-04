import einstein as E



class Sampler(object):
    def __init__(self):
        pass

    def sample(self):
        E.tools.AbstractMethod()


class GaussianSampler(Sampler):
    def __init__(self):
        super(GaussianSampler, self).__init__()
        
    
        
class SysSampler(GaussianSampler):
    def __init__(self):
        super(SysSampler, self).__init__()

    def sample(self, model_variances):
        """
        sigma_list contains sigma for each parameters
        """
        return E.tools.random.normal(0., model_variances)


class SuperSysSampler(GaussianSampler):
    def __init__(self):
        super(SuperSysSampler, self).__init__()

    def sample(self, sigma_list):
        """
        sigma_list contains sigma for each parameters
        """
        def abig(a):
            c1 = - 0.06655
            c2 = - 0.9706
            return E.tools.exp(c1 * (abs(a)**3 - abs(a)) / E.tools.log(abs(a)) + c2 * abs(a))
        def asmall(a):
            c3 = 0.124
            return E.ttols.exp(a)/(1.0 - a ** 3) ** (c3 *a)

        # normal sampling
        epsilon = E.tools.random.normal(0., sigma_list)
        theta = 0.67449 * sigma_list
        mirror_sigma_samples = E.tools.random.normal(0., theta)
        a = (theta -abs(epsilon)) / theta
        f_maps = [abig if x > 0 else asmall for x in a ]
        epsilon_star = E.tools.sign(epsilon) * theta * E.tools.array([v(x)  for v, x in zip(f_maps, a)])
        return epsilon, epsilon_star