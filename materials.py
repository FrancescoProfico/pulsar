import numpy as np

epsilon_tolerance_factor=1.0001

class Steel:
    def __init__(self, f_y, E, eps_u, E_hardening=0, law='ELASTIC-PERFECTLY_PLASTIC'):
        self.f_y=f_y
        self.name='S'+str(f_y)
        self.youngs_modulus=E
        self.eps_y=self.f_y/self.youngs_modulus
        self.eps_u=eps_u
        self.E_hardening=E_hardening
        self.law=law

    def sigma(self, eps):
        if self.law=='ELASTIC-PERFECTLY_PLASTIC':
            if abs(eps)<self.eps_y:
                return eps*self.youngs_modulus
            elif abs(eps)<self.eps_u*epsilon_tolerance_factor:
                return np.sign(eps)*self.f_y
            else:
                return 0
        elif self.law=='ELASTIC-PLASTIC_HARDENING':
            if abs(eps)<self.eps_y:
                return eps*self.youngs_modulus
            elif abs(eps)<self.eps_u*epsilon_tolerance_factor:
                return np.sign(eps)*(self.f_y+self.E_hardening*(abs(eps)-self.eps_y))
            else:
                return 0

    def failure_status(self, eps):
        if self.status_stage(eps)=='FAILED':
            return True
        else:
            return False

    def yielded_status(self, eps):
        if self.status_stage(eps)=='YIELDED':
            return True
        else:
            return False

    def status_stage(self, eps):
        if abs(eps) >= self.eps_u:
            return 'FAILED'
        elif abs(eps)>=self.eps_y:
            return 'YIELDED'
        else:
            return 'ELASTIC'

    def status_sign(self, eps):
        if eps>0:
            return 'TENSION'
        elif eps<0:
            return 'COMPRESSION'
        else:
            return 'UNLOADED'

    def general_status(self, eps):
        return self.status_stage(eps)+'-'+self.status_sign(eps)

    def plot_law(self, ax):
        max_eps=1.3*self.eps_u
        min_eps=-max_eps
        epss=np.arange(min_eps, max_eps, (max_eps-min_eps)/100)
        sigmas=[self.sigma(eps) for eps in epss]
        ax.plot(epss, sigmas, color='r')

class StructuralSteel(Steel):
    def __init__(self, f_y, E, eps_u, E_hardening=0, law='ELASTIC-PERFECTLY_PLASTIC'):
        super().__init__(f_y, E, eps_u, E_hardening=E_hardening, law=law)

class ReinforcementSteel(Steel):
    def __init__(self, f_y, E, eps_u, E_hardening=0, law='ELASTIC-PERFECTLY_PLASTIC'):
        super().__init__(f_y, E, eps_u, E_hardening=E_hardening, law=law)

class Concrete:
    def __init__(self, fck, law):
        self.name='Concrete '+r'$f_{ck}=$'+str(fck)+r'$ MPa$'
        gamma_c = 1.5
        alpha_cc = 0.85
        self.fck=fck
        self.law=law
        self.n=self.funct_n()
        self.fcd=self.fck/gamma_c*alpha_cc
        self.eps_c2=self.funct_eps_c2()
        self.eps_cu=self.funct_eps_cu2()

    def funct_fcm(self):
        return self.fck+8

    def funct_fctm(self):
        if self.fck<=50:
            return 0.3*self.fck**(2/3)
        else:
            fcm=self.funct_fcm()
            return 2.12*np.log(1+(fcm/10))

    def funct_Ecm(self):
        fcm = self.funct_fcm()
        return 22000*(fcm/10)**0.3

    def funct_eps_c2(self):
        if self.fck<50:
            return 2/1000
        else:
            return (2+0.085*(self.fck-50)**0.53)/1000

    def funct_eps_cu2(self):
        if self.fck<50:
            return 3.5/1000
        else:
            return (2.6+35*((90-self.fck)/100)**4)/1000

    def funct_n(self):
        if self.fck < 50:
            return 2
        else:
            return 1.4+23.4*((90-self.fck)/100)**4

    def sigma(self, eps):
        if self.law == 'PARABOLA-RECTANGLE':
            if eps > 0:
                return 0
            elif eps>=-self.eps_c2:
                return -self.fcd*(1-(1-abs(eps)/self.eps_c2)**self.n)
            elif eps>=-self.eps_cu*epsilon_tolerance_factor:
                return -self.fcd
            else:
                return 0

    def plot_law(self, ax):
        max_eps=1.3*self.eps_cu
        min_eps=-max_eps
        epss=np.arange(min_eps, max_eps, (max_eps-min_eps)/100)
        sigmas=[self.sigma(eps) for eps in epss]
        ax.plot(epss, sigmas, color='r')

    def failure_status(self, eps):
        if self.status_stage(eps)=='FAILED':
            return True
        else:
            return False

    def yielded_status(self, eps):
        if self.status_stage(eps)=='EPS_C2 REACHED':
            return True
        else:
            return False

    def status_stage(self, eps):
        if eps <= -self.eps_cu:
            return 'FAILED'
        elif eps<=-self.eps_c2:
            return 'EPS_C2 REACHED'
        elif eps<=0:
            return 'EPS_C2 NOT REACHED'
        else:
            return 'UNREAGENT'

    def status_sign(self, eps):
        if eps>0:
            return 'TENSION'
        elif eps<0:
            return 'COMPRESSION'
        else:
            return 'UNLOADED'

    def general_status(self, eps):
        return self.status_stage(eps)+'-'+self.status_sign(eps)



