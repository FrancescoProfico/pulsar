import numpy as np

class RectangularFibre:
    def __init__(self, material, y_max, y_min, b, name):
        self.name = name
        self.material=material
        self.y_max=y_max
        self.y_min=y_min
        self.b=b
        self.area=self.b*(self.y_max-self.y_min)
        self.y_mean=0.5*(self.y_max+self.y_min)

    def resultant_dN(self, eps_inf, eps_sup):
        if eps_inf*eps_sup<0:
            h=(self.y_max-self.y_min)
            x=h*abs(eps_sup)/(abs(eps_inf)+abs(eps_sup))
            dN=self.b*(x*0.5*self.material.sigma(eps_sup)+(h-x)*self.material.sigma(eps_inf))
        else:
            dN=self.area * 0.5 * (self.material.sigma(eps_inf) + self.material.sigma(eps_sup))

        return dN

    def resultant_dM(self, eps_inf, eps_sup, y_0):
        return self.resultant_dN(eps_inf=eps_inf, eps_sup=eps_sup)*(self.y_mean-y_0)

    def get_failure_status(self, eps_inf, eps_sup):
        fs = self.material.failure_status(eps_inf) or self.material.failure_status(eps_sup)
        return fs

    def get_yielding_status(self, eps_inf, eps_sup):
        ys = self.material.yielded_status(eps_inf) or self.material.yielded_status(eps_sup)
        return ys

    def draw(self, ax):
        ys = [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min]
        xs = [-self.b / 2, self.b / 2, self.b / 2, -self.b / 2, -self.b / 2]
        ax.plot(xs, ys, color='r', linewidth=0.2)

    def draw_sigmas(self, ax, deformative_state, closing_lines=0):
        if self.material.__class__.__name__=='StructuralSteel': k=0.1
        else: k=1

        if closing_lines==0:
            ys = [self.y_min, self.y_max]
        elif closing_lines==-1:
            ys = [self.y_min, self.y_min, self.y_max]
        elif closing_lines == 1:
            ys = [self.y_min, self.y_max, self.y_max]

        eps_t=deformative_state.get_eps(self.y_min)
        eps_b = deformative_state.get_eps(self.y_max)

        if closing_lines==0:
            sigmas=[self.material.sigma(eps_t), self.material.sigma(eps_b)]
        elif closing_lines==-1:
            sigmas = [0, self.material.sigma(eps_t), self.material.sigma(eps_b)]
        elif closing_lines == 1:
            sigmas = [self.material.sigma(eps_t), self.material.sigma(eps_b), 0]

        sigmas=[k*s for s in sigmas]
        ax.plot(sigmas, ys, color='k', linewidth=0.5)

class ReinforcementFibre:
    def __init__(self, area, y, material):
        self.area=area
        self.y=y
        self.y_mean=self.y
        self.y_max = self.y
        self.y_min = self.y
        self.material=material

    def resultant_dN(self, eps_inf, eps_sup):
        return self.area * 0.5*(self.material.sigma(eps_inf)+self.material.sigma(eps_sup))

    def resultant_dM(self,  eps_inf, eps_sup, y_0):
        return self.resultant_dN(eps_inf=eps_inf, eps_sup=eps_sup) * (self.y - y_0)

    def get_failure_status(self, eps_inf, eps_sup):
        fs=self.material.failure_status(eps_inf) or self.material.failure_status(eps_sup)
        return fs

    def get_yielding_status(self, eps_inf, eps_sup):
        ys = self.material.yielded_status(eps_inf) or self.material.yielded_status(eps_sup)
        return ys

    def draw(self, ax):
        xmin, xmax=ax.get_xlim()
        w=(xmax-xmin)
        ax.plot([-w/10, w/10], [self.y, self.y], color='k', linewidth=0.5)

    def draw_sigmas(self, ax, deformative_state):
        if self.material.__class__.__name__=='ReinforcementSteel': k=0.1
        else: k=1
        ys = [self.y, self.y]
        eps = deformative_state.get_eps(self.y)
        sigmas = [0, self.material.sigma(eps)]
        sigmas = [k * s for s in sigmas]
        ax.plot(sigmas, ys, color='k', linewidth=2)
