from fibres import RectangularFibre, ReinforcementFibre
import numpy as np
import drawing_tools
from matplotlib.patches import Rectangle
import plot_setup
from plot_setup import annotations_fontsize
import matplotlib.pyplot as plt
import xlwings as xw

def xls_search(designation, key):
    with xw.App(visible=False) as app:
        book = xw.Book("sections_db/Sections and Merchant Bars-ArcelorMittal.xlsx")
        sh=book.sheets["EN sections"]
        ref_cell = sh["B4"]
        rownum = 850
        colnum = 30

        j=0
        correspondence_column = False
        while correspondence_column is False and j<=colnum:
            j+=1
            if ref_cell.offset(0, j).value == key:
                correspondence_column=True
        if correspondence_column is False:
            print('Column not found')

        i=0
        correspondence_row=False
        while correspondence_row is False and i<=rownum:
            i+=1
            if ref_cell.offset(i,0).value==designation:
                correspondence_row=True
                value=ref_cell.offset(i, j).value
        if correspondence_row is False:
            print('Row not found')

        book.close()
        return value

class Section:
    def __init__(self, name, parts):
        self.name=name
        self.parts=parts

        self.y_min=self.get_y_min()
        self.y_max = self.get_y_max()
        self.halfheight=self.get_halfheight()
        self.events=[]

    def failure_status(self):
        failure = False
        for p in self.parts:
            for f in p.fibres:
                eps_sup = p.deformative_state.get_eps(f.y_min)
                eps_inf = p.deformative_state.get_eps(f.y_max)
                if f.get_failure_status(eps_inf=eps_inf, eps_sup=eps_sup) is True:
                    failure = True
        return failure

    def update_status(self, status):
        def my_append(a,b):
            if a=='':
                return b
            else:
                return a+'; '+b
        description=''
        for p in self.parts:
            for f in p.fibres:
                eps_sup = p.deformative_state.get_eps(f.y_min)
                eps_inf = p.deformative_state.get_eps(f.y_max)
                if f.get_failure_status(eps_inf=eps_inf, eps_sup=eps_sup) is True:
                    if f.material.__class__.__name__=='Concrete':
                        if status.concrete_failure is False:
                            status.concrete_failure = True
                            description=my_append(description,'Concrete crushing')
                    elif f.material.__class__.__name__=='StructuralSteel':
                        if status.steel_failure is False:
                            status.steel_failure = True
                            description = my_append(description, 'Structural steel failure')
                    elif f.material.__class__.__name__=='ReinforcementSteel':
                        if status.reinforcement_failure is False:
                            status.reinforcement_failure = True
                            description = my_append(description, 'Reinforcement steel failure')
                if f.get_yielding_status(eps_inf=eps_inf, eps_sup=eps_sup) is True:
                    if f.material.__class__.__name__=='Concrete':
                        if status.concrete_yielded is False:
                            status.concrete_yielded = True
                            description = my_append(description, 'Concrete reaching eps_c2')
                    elif f.material.__class__.__name__=='StructuralSteel':
                        if status.steel_yielded is False:
                            status.steel_yielded = True
                            description = my_append(description, 'Structural steel yielding')
                    elif f.material.__class__.__name__=='ReinforcementSteel':
                        if status.reinforcement_yielded is False:
                            status.reinforcement_yielded = True
                            description = my_append(description, 'Reinforcement steel failure')
        return description

    def get_max_width(self):
        maxw=0
        for p in self.parts:
            if p.__class__.__name__!='ReinforcementPart':
                maxw=max(maxw, p.get_max_width())
        return maxw

    def get_halfheight(self):
        return 0.5*(self.y_max+self.y_min)

    def get_y_min(self):
        for n, p in enumerate(self.parts):
            if n==0:
                y_min=p.get_y_min()
            else:
                y_min=np.min([y_min, p.get_y_min()])
        return y_min

    def get_y_max(self):
        for n, p in enumerate(self.parts):
            if n==0:
                y_max=p.get_y_max()
            else:
                y_max=np.max([y_max, p.get_y_max()])
        return y_max

    def get_materials_list(self):
        materials_list = []
        for p in self.parts:
            if len(materials_list) != 0:
                for m in materials_list:
                    if p.material is not m:
                        materials_list.append(p.material)
            else:
                materials_list.append(p.material)
        return materials_list

    def draw_section(self, ax, annotate=False):
        for p in self.parts:
            p.draw(ax)
            if annotate is True:
                if p.material.__class__.__name__=='ReinforcementSteel':
                    for r in p.geometry_description:
                        r.annotate_plot(ax)
                else:
                    p.geometry_description.annotate_plot(ax)

    def draw_halfheight_line(self, ax):
        xmin, xmax = ax.get_xlim()
        y = self.halfheight
        ax.plot([xmin, xmax], [y, y], color=[0.8,0.8,0.85], linestyle='-.', linewidth=1, zorder=0)

    def draw_fibres(self, ax):
        for p in self.parts: p.draw_fibres(ax)

    def draw_deformative_state(self, ax):
        for p in self.parts:
            p.draw_deformative_state(ax)

    def draw_ultimate_strains(self, ax):
        for p in self.parts:
            p.draw_ultimate_strains(ax)

    def draw_sigmas_of_defstate(self, ax):
        for p in self.parts:
            p.draw_sigmas_of_defstate(ax)

    def draw_partial_resultants(self, ax):
        for p in self.parts:
            p.draw_partial_resultant(ax)

    def draw_global_resultants(self, ax):
        N=self.get_resultant_N()
        M = self.get_resultant_M(y_0=self.halfheight)
        drawing_tools.draw_global_resultant_arrow(ax, y=self.halfheight, value=N)
        drawing_tools.draw_moment_resultant_arrow(ax, y=self.halfheight, value=M)

    def draw_neutral_axis(self, ax, bool_string=True):
        for p in self.parts:
            p.draw_neutral_axis(ax, bool_string)

    def get_resultant_N(self):
        N=0
        for p in self.parts: N+=p.get_resultant_N()
        return N

    def get_resultant_N_concrete(self):
        N=0
        for p in self.parts:
            if p.material.__class__.__name__=='Concrete':
                N+=p.get_resultant_N()
        return N

    def get_resultant_N_structural_steel(self):
        N=0
        for p in self.parts:
            if p.material.__class__.__name__=='StructuralSteel':
                N+=p.get_resultant_N()
        return N

    def get_resultant_M(self, y_0):
        M = 0
        for p in self.parts: M += p.get_resultant_M(y_0=y_0)
        return M

    def moment_curvature_diagram(self, ax, axial_force=0, slip_strain_ratio=None):
        if slip_strain_ratio is None: slip_strain_ratio=1

        status=Status()
        curvature=0
        failure=False
        increment_curvature=1*10**-6
        max_iter=1000
        tolerance_N=0.1
        N=0
        step=0
        eps_0 = 0
        while failure is False:
            step+=1
            # eps_0=0
            convergence=False
            iteration=0
            max_iter_condition=False
            eps_increment = -0.0001
            N_previous=0
            while convergence is False and max_iter_condition is False and failure is False:
                # print('Iteration: {}'.format(iteration))
                ds_concrete=DeformativeState(curvature=curvature, eps_0=eps_0*slip_strain_ratio)
                ds_steel = DeformativeState(curvature=curvature, eps_0=eps_0)
                self.set_def_state(ds_concrete, ds_steel)

                N=self.get_resultant_N()
                residual=N-axial_force
                if abs(residual)<=tolerance_N:
                    convergence=True
                    M=self.get_resultant_M(y_0=self.halfheight)
                else:
                    if N*N_previous<0: eps_increment=-eps_increment/2
                    N_previous = N
                    eps_0=eps_0+eps_increment
                    iteration += 1
                    if iteration >= max_iter:
                        max_iter_condition = True
                        print('Max iter number reached')
                        M = 0

            description=self.update_status(status)
            if description!='':
                ev=Event(section=self, label=len(self.events)+1, description=description, def_state_concrete=ds_concrete, def_state_structural_steel=ds_steel, plotting_list=['moment_curvature'])
                self.events.append(ev)

            new_row=np.array([[step, curvature, eps_0, residual, N, M, convergence]])
            if step==1: table=new_row
            else: table=np.append(table, new_row, axis=0)

            failure=self.failure_status()
            if failure is False:
                curvature += increment_curvature

            print('Step: {}; Convergence: {}; Residual: {}; Iterations: {}; Failure: {}; Curvature: {}'.format(step, convergence, residual, iteration, failure, curvature))

        ax.plot(table[:,1], table[:,5]/10**6, color='k', linewidth=0.5)

        for ev in self.events:
            ev.plot_on_graph(ax, 'moment_curvature')

    def plot_N_y(self,slip_strain):
        a=np.arange(self.y_min, self.y_max, (self.y_max-self.y_min)/10000)
        print(a)
        for n, pna in enumerate(a):
            ds_steel, ds_concrete=self.uls_deformative_state(pna, slip_strain, 'POS')
            self.set_def_state(ds_concrete, ds_steel)
            n_resultant=self.get_resultant_N_concrete()
            ult_curv=ds_steel.curvature
            new_row=np.array([[pna, n_resultant, ult_curv]])
            if n==0:
                table=new_row
            else:
                table=np.append(table, new_row, axis=0)

        fig, [ax1, ax2]=plt.subplots(1,2)
        ax1.plot(table[:,0], table[:,1], color='b')
        ax2.plot(table[:, 0], table[:, 2], color='b')
        plt.show()

    def uls_deformative_state(self, neutral_axis_y, slip_strain, curv_mode):
        if curv_mode=='POS': coeff=1; func=min
        else: coeff=-1; func=max

        ultimate_curvature=coeff*1
        for p in self.parts:
            for n,f in enumerate([p.fibres[0], p.fibres[-1]]):
                if abs((f.y_mean - neutral_axis_y))!=0:
                    if f.material.__class__.__name__ == 'ReinforcementSteel':
                        ultimate_curvature = func(coeff * (f.material.eps_u + slip_strain) / abs((f.y_mean - neutral_axis_y)), ultimate_curvature)
                    else:
                        if n == 0: y_ref = f.y_min
                        elif n == 1: y_ref = f.y_max
                        if f.material.__class__.__name__=='Concrete':
                            if coeff*(y_ref - neutral_axis_y)*(slip_strain+f.material.eps_cu)/(f.material.eps_cu)<=0:
                                ultimate_curvature=func(coeff*(f.material.eps_cu+slip_strain) / abs((y_ref - neutral_axis_y)),ultimate_curvature)
                        if f.material.__class__.__name__=='StructuralSteel':
                            ultimate_curvature=func(coeff*f.material.eps_u / abs((y_ref - neutral_axis_y)),ultimate_curvature)

        return DeformativeState(eps_0=-ultimate_curvature*neutral_axis_y, curvature=ultimate_curvature), DeformativeState(eps_0=-ultimate_curvature*neutral_axis_y+slip_strain, curvature=ultimate_curvature)

    def moment_axialforce_interaction_diagram(self, ax1, ax2, ax3, slip_strain):
        vals=np.array([-10000])
        vals=np.append(vals,np.arange(-100+self.y_min, self.y_max+100, 5), 0)
        vals = np.append(vals, [10000], 0)
        for curv_mode in ['POS', 'NEG']:
            for n, neutral_axis_structural_steel in enumerate(vals):
                ds_steel, ds_concrete=self.uls_deformative_state(neutral_axis_structural_steel, slip_strain, curv_mode)
                ultimate_curvature=ds_steel.curvature
                self.set_def_state(ds_concrete, ds_steel)
                N=self.get_resultant_N()
                M=self.get_resultant_M(y_0=self.halfheight)
                new_row = np.array([[n, neutral_axis_structural_steel, N, M, ultimate_curvature]])
                if n==0:
                    table=new_row
                else:
                    table=np.append(table, new_row, axis=0)

                if (n == 0 or n == len(vals) - 1) and curv_mode=='POS':
                    ev = Event(section=self, label=len(self.events) + 1, description='A',
                               def_state_concrete=ds_concrete,
                               def_state_structural_steel=ds_steel, plotting_list=['moment_axialforce_interaction_diagram'])
                    self.events.append(ev)


            ax1.plot(table[:,2]/10**3, table[:,3]/10**6, color='k', linewidth=0.5)
            ax2.plot(table[:, 1], table[:, 2] / 10 ** 3, color='r', linewidth=0.5)
            ax3.plot(table[:, 1], table[:, 4], color='b', linewidth=0.5)

            nth=int(round(len(vals)/2+1,0))
            if curv_mode=='POS':
                for ax in [ax1]:
                    t = ax.text(table[nth,2]/10**3, table[nth,3]/10**6, r'$\epsilon_{slip}=$' + str(round(slip_strain*1000, 1)) + r'$^o/_{oo}$', color='k',
                                fontsize=annotations_fontsize, va='center')
                    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', boxstyle='square, pad=0.0'))

        for ev in self.events:
            ev.plot_on_graph(ax1, 'moment_axialforce_interaction_diagram')


    def uls(self, axial_force, slip_strain):
        convergence=False
        tolerance=1000
        neutral_axis_y=-self.halfheight*2
        step_neutral_axis=self.halfheight/11
        residual_previous=1
        max_iter=10000

        iter=0
        while convergence is False and iter<=max_iter:
            iter+=1
            ds_steel, ds_concrete=self.uls_deformative_state(neutral_axis_y=neutral_axis_y, slip_strain=slip_strain, curv_mode='POS')
            self.set_def_state(ds_concrete, ds_steel)
            residual=self.get_resultant_N()-axial_force
            if abs(residual)<=tolerance:
                convergence=True
                print('Converged!')
            else:
                if residual_previous*residual<0 and iter>1:
                    step_neutral_axis=-step_neutral_axis/3
                neutral_axis_y+=step_neutral_axis
                residual_previous=np.sign(residual)
            print('Residual={}, PNA={}, step={}'.format(residual, neutral_axis_y, step_neutral_axis))

        if convergence is True:
            return ds_steel, ds_concrete
        else: print('Max iter number reached')

    def uls_fsi(self, axial_force):
        ds_steel, ds_concrete=self.uls(axial_force=axial_force, slip_strain=0)
        ev = Event(section=self, label=len(self.events) + 1, description='ULS-FSI', def_state_concrete=ds_concrete,
                   def_state_structural_steel=ds_steel, plotting_list=['partial_shear_diagram', 'moment_axialforce_interaction_diagram', 'moment_curvature'])
        self.events.append(ev)

    def partial_shear_diagram(self, ax, axial_force=0):
        step=0.02
        slip_strains=np.arange(0,1+step,step)
        slip_strains=[0.1*ss**3 for ss in slip_strains]
        for n, ss in enumerate(slip_strains):
            ds_steel, ds_concrete = self.uls(axial_force, ss)
            self.set_def_state(ds_concrete=ds_concrete, ds_steel=ds_steel)
            M = self.get_resultant_M(y_0=self.halfheight)
            Nc=self.get_resultant_N_concrete()
            new_row=np.array([[Nc, M]])
            if n==0:
                table=new_row
            else:
                table=np.append(table, new_row, axis=0)

            if n==len(slip_strains)-1:
                ev = Event(section=self, label=len(self.events) + 1, description='No Shear Interaction',
                           def_state_concrete=ds_concrete,
                           def_state_structural_steel=ds_steel, plotting_list=['partial_shear_diagram','moment_axialforce_interaction_diagram'])
                self.events.append(ev)

        ax.plot(-table[:, 0]/10**3, table[:, 1]/10**6, color='k', linewidth=0.5)

        for ev in self.events:
            ev.plot_on_graph(ax, 'partial_shear_diagram')

    def set_def_state(self, ds_concrete, ds_steel):
        for p in self.parts:
            if p.material.__class__.__name__ == 'Concrete':
                p.set_def_state(ds_concrete)
            elif p.material.__class__.__name__ == 'StructuralSteel':
                p.set_def_state(ds_steel)
            elif p.material.__class__.__name__ == 'ReinforcementSteel':
                p.set_def_state(ds_concrete)

    def plot_section(self, directory):
        cm = 1 / 2.54
        fig_section_geometry, ax = plt.subplots(1, 1, figsize=(10 * cm, 10 * cm), constrained_layout=True)
        plot_setup.section_chart_setup(ax)
        self.draw_section(ax, annotate=True)
        self.draw_halfheight_line(ax)
        self.draw_fibres(ax)
        ax.set_title(self.name)
        plt.savefig(directory + 'section.pdf')

    def plot_def_state(self, directory, filename, color=None, title=None):
        if color is None:
            color='w'
            text_color='k'
        else: text_color='w'
        if title is None:
            title='Deformed state'

        cm=1/2.54
        fig_section, [ax11, ax12, ax13, ax14, ax15] = plt.subplots(1, 5, figsize=(25 * cm, 5 * cm),
                                                              gridspec_kw={'width_ratios': [4, 1, 1, 3, 3]})

        t=fig_section.suptitle('     '+title, fontweight='bold', color=text_color, ha='left', x=0.0, y=1, va='top')
        t.set_bbox(dict(facecolor=color, edgecolor='w', alpha=1, boxstyle='square, pad=0.1'))

        fig_section.tight_layout(pad=1)
        plot_setup.section_chart_setup(ax11)
        plot_setup.deformative_state_chart_setup(ax12, self)
        plot_setup.stress_state_chart_setup(ax13, self)
        plot_setup.partial_resultants_chart_setup(ax14, self)
        plot_setup.global_resultants_chart_setup(ax15, self)

        # s.draw_fibres(ax11)
        self.draw_neutral_axis(ax11)
        self.draw_section(ax11)
        self.draw_deformative_state(ax12)
        self.draw_ultimate_strains(ax12)
        self.draw_neutral_axis(ax12, bool_string=False)
        self.draw_sigmas_of_defstate(ax13)
        self.draw_neutral_axis(ax13, bool_string=False)
        self.draw_partial_resultants(ax14)
        self.draw_neutral_axis(ax14, bool_string=False)
        self.draw_global_resultants(ax15)
        self.draw_neutral_axis(ax15, bool_string=False)
        for ax in fig_section.get_axes():
            self.draw_halfheight_line(ax)

        ax11.sharey(ax12)
        ax13.sharey(ax11)

        w=self.y_max-self.y_min
        for ax in fig_section.get_axes():
            ax.set_ylim(self.y_max+w/10, self.y_min-w/10)


        plt.savefig(directory + filename+'-deformative_state.pdf')

    def plot_partial_shear_diagram(self, directory):
        cm=1/2.54
        fig_partial_shear_diagram, ax41 = plt.subplots(1, 1, figsize=(15 * cm, 10 * cm))
        plot_setup.partial_shear_diagram_chart_setup(ax41)
        self.partial_shear_diagram(ax41, axial_force=0)

        ax41.set_xlim(left=0)
        ax41.set_ylim(bottom=0)
        plt.savefig(directory + 'partial_shear_diagram.pdf')

    def plot_moment_axialforce_interaction_diagram(self, directory):
        cm=1/2.54
        fig_moment_axialforce_interaction_diagram, [ax31, ax32, ax33] = plt.subplots(1, 3, figsize=(25 * cm, 10 * cm))
        fig_moment_axialforce_interaction_diagram.tight_layout(pad=3)
        fig_moment_axialforce_interaction_diagram.suptitle("M-N INTERACTION", fontweight='bold')
        plot_setup.moment_axialforce_interaction_chart_setup(ax31)
        plot_setup.neutralaxis_axialforce_chart_setup(ax32, section=self)
        plot_setup.neutralaxis_ultcurvature_chart_setup(ax33, section=self)

        for ss in [0,0.1]:
            self.moment_axialforce_interaction_diagram(ax31, ax32, ax33, slip_strain=ss)

        drawing_tools.draw_axis(ax31)
        plt.savefig(directory + 'moment_axialforce_interaction_diagram.pdf')

    def plot_moment_curvature_diagram(self, directory):
        cm=1/2.54
        fig_moment_curvature_diagram, ax21=plt.subplots(1, 1, figsize=(15 * cm, 10 * cm))
        plot_setup.moment_curvature_chart_setup(ax21)
        self.moment_curvature_diagram(ax21, axial_force=0, slip_strain_ratio=1)
        ax21.set_xlim(left=0)
        ax21.set_ylim(bottom=0)
        plt.savefig(directory + 'moment_curvature_diagram.pdf')

    def plot_materials(self, directory):
        cm=1/2.54
        materials_list=self.get_materials_list()

        fig_materials = plt.figure()
        for m in materials_list:
            ax = fig_materials.add_axes([0, 0, 1, 1])
            plot_setup.material_law_chart_setup(ax, m)
            ax.plot_law(ax)

        plt.savefig(directory + 'materials.pdf')

    def plot_events_def_state(self, directory):
        for ev in self.events:
            ev.plot_def_state(directory=directory)

class DeformativeState:
    def __init__(self, eps_0, curvature):
        self.eps_0=eps_0
        self.curvature=curvature

    def get_eps(self, y):
        return self.eps_0+y*self.curvature

    def get_neutral_axis(self):
        if self.curvature!=0:
            return -self.eps_0/self.curvature
        else: return 0

class Part:
    def __init__(self, name, geometry_description, material, deformative_state):
        self.name = name
        self.material = material
        self.geometry_description = geometry_description
        self.fibres=[]
        if deformative_state is None:
            self.deformative_state=DeformativeState(0,0)
        else:
            self.deformative_state=deformative_state

    def set_def_state(self, def_state):
        self.deformative_state=def_state

    def get_resultant_N(self, sign=None):
        N=0
        if sign is None:
            for f in self.fibres:
                N+=f.resultant_dN(eps_inf=self.deformative_state.get_eps(f.y_max),eps_sup=self.deformative_state.get_eps(f.y_min))
        elif sign=='POSITIVE':
            for f in self.fibres:
                dN=f.resultant_dN(eps_inf=self.deformative_state.get_eps(f.y_max),eps_sup=self.deformative_state.get_eps(f.y_min))
                if dN>0: N+=dN
        elif sign=='NEGATIVE':
            for f in self.fibres:
                dN = f.resultant_dN(eps_inf=self.deformative_state.get_eps(f.y_max),eps_sup=self.deformative_state.get_eps(f.y_min))
                if dN < 0: N += dN
        return N

    def get_resultant_M(self,y_0, sign=None):
        M = 0
        if sign is None:
            for f in self.fibres:
                M += f.resultant_dM(eps_inf=self.deformative_state.get_eps(f.y_max),eps_sup=self.deformative_state.get_eps(f.y_min), y_0=y_0)
        elif sign=='POSITIVE':
            for f in self.fibres:
                dN=f.resultant_dN(eps_inf=self.deformative_state.get_eps(f.y_max),eps_sup=self.deformative_state.get_eps(f.y_min))
                if dN>0: M+=f.resultant_dM(eps_inf=self.deformative_state.get_eps(f.y_max),eps_sup=self.deformative_state.get_eps(f.y_min), y_0=y_0)
        elif sign=='NEGATIVE':
            for f in self.fibres:
                dN = f.resultant_dN(eps_inf=self.deformative_state.get_eps(f.y_max),eps_sup=self.deformative_state.get_eps(f.y_min))
                if dN < 0: M += f.resultant_dM(eps_inf=self.deformative_state.get_eps(f.y_max),eps_sup=self.deformative_state.get_eps(f.y_min), y_0=y_0)
        return M

    def get_resultant_y(self, y_0, sign=None):
        M=self.get_resultant_M(y_0=y_0, sign=sign)
        N=self.get_resultant_N(sign=sign)
        if N!=0:
            return M/N
        else:
            return 0

    def draw_neutral_axis(self, ax, bool_string):
        xmin, xmax=ax.get_xlim()

        y=self.deformative_state.get_neutral_axis()
        ax.plot([xmin, xmax],[y, y], color='m', linestyle='-.', linewidth=1)

        if bool_string is True:
            t = ax.text(xmin+0.7*(xmax-xmin), y, 'y=' + str(round(y, 1)) + ' mm', color='m', fontsize=annotations_fontsize, va='center')
            t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', boxstyle='square, pad=0.1'))

    def get_yielding_status(self):
        for f in [self.fibres[0], self.fibres[-1]]:
            status = f.get_yielding_status(eps_sup=self.deformative_state.get_eps(y=f.y_min),
                                  eps_inf=self.deformative_state.get_eps(y=f.y_max))
        return status

    def get_failure_status(self):
        for f in [self.fibres[0], self.fibres[-1]]:
            status = f.get_failure_status(eps_sup=self.deformative_state.get_eps(y=f.y_min),
                                  eps_inf=self.deformative_state.get_eps(y=f.y_max))
        return status

class ReinforcementPart(Part):
    def __init__(self, name, geometry_description, material, deformative_state=None):
        super().__init__(name=name, geometry_description=geometry_description, material=material, deformative_state=deformative_state)
        self.fibres=self.set_fibres()

    def get_y_min(self):
        for n, r in enumerate(self.geometry_description):
            if n==0:
                y_min=r.y
            else:
                y_min=np.min([y_min, r.y])
        return y_min

    def get_y_max(self):
        for n, r in enumerate(self.geometry_description):
            if n==0:
                y_max=r.y
            else:
                y_max=np.max([y_max, r.y])
        return y_max

    def set_fibres(self):
        fibres = []
        for gd in self.geometry_description:
            fibres.append(ReinforcementFibre(area=gd.area, material=self.material, y=gd.y))
        return fibres

    def draw(self, ax):
        for f in self.fibres:
            f.draw(ax)

    def draw_fibres(self,ax):
        for f in self.fibres:
            f.draw(ax)

    def draw_deformative_state(self, ax):
        for f in self.fibres:
            ys=[f.y, f.y]
            epss=[0, self.deformative_state.get_eps(f.y)]
            ax.plot(epss, ys, color='k', linewidth=0.5)

    def draw_ultimate_strains(self, ax):
        xmin, xmax = ax.get_xlim()

        eps=self.material.eps_u
        ys=[self.get_y_min(), self.get_y_min()]
        epss=[-eps, -eps]

        ax.plot(epss, ys, color='r', linewidth=0.5, linestyle='--')
        epss=[-e for e in epss]
        ax.plot(epss, ys, color='r', linewidth=0.5, linestyle='--')

        ax.set_xlim(xmin, xmax)

    def draw_sigmas_of_defstate(self, ax):
        for f in self.fibres:

            f.draw_sigmas(ax, deformative_state=self.deformative_state)

    def draw_partial_resultant(self, ax):
        for sign in ['POSITIVE', 'NEGATIVE']:
            y=self.get_resultant_y(0, sign)
            N = self.get_resultant_N(sign)
            if N!=0: drawing_tools.draw_partial_resultant_arrow(ax, y=y, value=N, string=self.name)

class FlatPart(Part):
    def __init__(self, name, geometry_description, material, deformative_state=None):
        super().__init__(name=name, geometry_description=geometry_description, material=material, deformative_state=deformative_state)
        self.fibres = self.set_fibres(dy=2)
        self.y_min=self.get_y_min()
        self.y_max = self.get_y_max()

    def get_max_width(self):
        maxw=0
        for ep in self.geometry_description.elementary_parts: maxw=max(ep.b, maxw)
        return maxw

    def get_y_min(self):
        for n, ep in enumerate(self.geometry_description.elementary_parts):
            if n==0:
                y_min=ep.y_min
            else:
                y_min=np.min([y_min, ep.y_min])
        return y_min

    def get_y_max(self):
        for n, ep in enumerate(self.geometry_description.elementary_parts):
            if n==0:
                y_max=ep.y_max
            else:
                y_max=np.max([y_max, ep.y_max])
        return y_max

    def set_fibres(self, dy):
        fibres = []
        i = 0
        for ep in self.geometry_description.elementary_parts:
            i += 1
            j=0
            while j * dy + ep.y_min < ep.y_max:
                j+=1
                y_max_fibre = min(j * dy + ep.y_min, ep.y_max)
                y_min_fibre = (j-1) * dy + ep.y_min
                fibres.append(RectangularFibre(b=ep.b, y_max=y_max_fibre, y_min=y_min_fibre, material=self.material, name=self.name+'-'+str(i)+'-'+str(j)))
        return fibres

    def draw(self, ax):
        for ep in self.geometry_description.elementary_parts:
            if self.material.__class__.__name__=='StructuralSteel':
                filled=True
            else:
                filled=False
            ep.draw(ax, filled)

    def draw_fibres(self, ax):
        for f in self.fibres:
            f.draw(ax)

    def draw_deformative_state(self, ax):
        ys=[self.y_min, self.y_min, self.y_max, self.y_max]
        eps_1=self.deformative_state.get_eps(ys[1])
        eps_2=self.deformative_state.get_eps(ys[2])
        epss=[0, eps_1, eps_2, 0]
        labels=['', str(round(eps_1*1000,2))+r'$^o/_{oo}$', str(round(eps_2*1000,2))+r'$^o/_{oo}$', '']
        ax.plot(epss, ys, color='k', linewidth=0.5)

        for indx in [1,2]:
            if epss[indx]>0:
                ha='left'
            else:
                ha='right'
            t = ax.text(epss[indx], ys[indx], labels[indx], color='k', fontsize=annotations_fontsize, va='center', ha=ha)
            t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', boxstyle='square, pad=0'))

    def draw_ultimate_strains(self, ax):
        xmin, xmax=ax.get_xlim()

        if self.material.__class__.__name__=='Concrete':
           eps=self.material.eps_cu
        elif self.material.__class__.__name__=='StructuralSteel':
            eps = self.material.eps_u

        ys=[self.y_min, self.y_max]
        epss=[-eps, -eps]

        ax.plot(epss, ys, color='r', linewidth=0.5, linestyle='--')
        if self.material.__class__.__name__ == 'StructuralSteel':
            epss=[-e for e in epss]
            ax.plot(epss, ys, color='r', linewidth=0.5, linestyle='--')

        ax.set_xlim(xmin, xmax)


    def draw_sigmas_of_defstate(self, ax):
        for n, f in enumerate(self.fibres):
            if n==0:
                closing_lines=-1
            elif n == len(self.fibres)-1:
                closing_lines = 1
            else:
                closing_lines = 0

            f.draw_sigmas(ax, deformative_state=self.deformative_state, closing_lines=closing_lines)

    def draw_partial_resultant(self, ax):
        for sign in ['POSITIVE', 'NEGATIVE']:
            y=self.get_resultant_y(0, sign)
            N = self.get_resultant_N(sign)
            if N!=0: drawing_tools.draw_partial_resultant_arrow(ax, y=y, value=N, string=self.name)

class Geometry:
    def __init__(self):
        self.elementary_parts=[]

    def get_area(self):
        area=0
        for ep in self.elementary_parts:
            area+=ep.get_area()
        return area

    def get_static_moment(self):
        sm = 0
        for ep in self.elementary_parts:
            sm += ep.get_static_moment()
        return sm

    def get_centroid(self):
        return self.get_static_moment()/self.get_area()

class ReinforcementGeometry(Geometry):
    def __init__(self, area, y):
        Geometry.__init__(self)
        self.area=area
        self.y=y

    def annotate_plot(self, ax):
        offset=20
        drawing_tools.write_measure(ax, 200, self.y, 'left', r'$A_s$', self.area, r'$mm^2$')

class RectangleGeometry(Geometry):
    def __init__(self, b, h, y_min):
        Geometry.__init__(self)
        self.y_min=y_min
        self.b=b
        self.h=h
        self.elementary_parts=self.set_elementary_parts()

    def set_elementary_parts(self):
        y_max=self.y_min+self.h
        y_min=self.y_min
        return [RectangularPart(y_max, y_min, self.b)]

    def annotate_plot(self, ax):
        offset=20
        drawing_tools.write_measure(ax, 0, self.y_min-offset, 'center', r'$B$', self.b, 'mm')
        drawing_tools.write_measure(ax, 0, self.y_min + self.h/2, 'center', r'$h$', self.h, 'mm')

class DoubleSymmDoubleTGeometry(Geometry):
    def __init__(self, b, h, t_w, t_f, r, y_min):
        Geometry.__init__(self)
        self.y_min=y_min
        self.b=b
        self.h=h
        self.t_w=t_w
        self.t_f=t_f
        self.r=r
        self.elementary_parts = self.set_elementary_parts()

    def annotate_plot(self, ax):
        offset=20
        drawing_tools.write_measure(ax, self.b / 2 + offset, self.y_min + self.h - self.t_f / 2, 'left', r'$t_f$', self.t_f, 'mm')
        drawing_tools.write_measure(ax, 0, self.y_min + self.h + offset, 'center', r'$b$', self.b, 'mm')
        drawing_tools.write_measure(ax, self.t_w + offset, self.y_min + (self.h - self.t_f) / 2, 'left', r'$t_w$', self.t_w, 'mm')

    def set_elementary_parts(self):
        y_max=self.y_min+self.h
        y_min=self.y_min

        t_flange=RectangularPart(y_max=y_min+self.t_f, y_min=y_min, b=self.b)
        web=RectangularPart(y_max=y_max-self.t_f, y_min=y_min+self.t_f, b=self.t_w)
        b_flange=RectangularPart(y_max=y_max, y_min=y_max-self.t_f, b=self.b)
        return [t_flange, web, b_flange]

class HotRolledSection(DoubleSymmDoubleTGeometry):
    def __init__(self, designation, y_min):
        h=xls_search(designation, 'h')
        t_f = xls_search(designation, 'tf')
        b = xls_search(designation, 'b')
        t_w = xls_search(designation, 'tw')
        r = xls_search(designation, 'r')
        DoubleSymmDoubleTGeometry.__init__(self,b, h, t_w, t_f, r, y_min)

class SingleSymmDoubleTGeometry(Geometry):
    def __init__(self, b_t, b_b, h, t_w, t_ft, t_fb, y_min):
        Geometry.__init__(self)
        self.y_min=y_min
        self.b_t=b_t
        self.b_b = b_b
        self.h=h
        self.t_w=t_w
        self.t_ft=t_ft
        self.t_fb = t_fb
        self.elementary_parts = self.set_elementary_parts()

    def set_elementary_parts(self):
        y_max = self.y_min + self.h
        y_min = self.y_min

        t_flange = RectangularPart(y_max=y_min + self.t_ft, y_min=y_min, b=self.b_t)
        web = RectangularPart(y_max=y_max - self.t_fb, y_min=y_min + self.t_ft, b=self.t_w)
        b_flange = RectangularPart(y_max=y_max, y_min=y_max - self.t_fb, b=self.b_b)
        return [t_flange, web, b_flange]

class SingleTGeometry(Geometry):
    def __init__(self, b, h_singleT, t_w, t_f, y_min):
        Geometry.__init__(self)
        self.y_min = y_min
        self.b=b
        self.h_singleT=h_singleT
        self.t_w=t_w
        self.t_f=t_f
        self.elementary_parts = self.set_elementary_parts()

    def set_elementary_parts(self):
        y_max = self.y_min + self.h_singleT
        y_min = self.y_min

        web = RectangularPart(y_max=y_max - self.t_f, y_min=y_min, b=self.t_w)
        b_flange = RectangularPart(y_max=y_max, y_min=y_max - self.t_f, b=self.b)
        return [web, b_flange]

    def annotate_plot(self, ax):
        offset=20

        drawing_tools.write_measure(ax, self.b/2+offset, self.y_min+self.h_singleT-self.t_f/2, 'left', r'$t_f$', self.t_f, 'mm')
        drawing_tools.write_measure(ax, 0, self.y_min + self.h_singleT+offset, 'center', r'$b$', self.b, 'mm')
        drawing_tools.write_measure(ax, self.t_w+offset, self.y_min + (self.h_singleT-self.t_f)/2, 'left', r'$t_w$', self.t_w, 'mm')

class RectangularPart:
    nr_rectangular_parts=0
    def __init__(self, y_max, y_min, b):
        RectangularPart.nr_rectangular_parts+=1
        self.id=RectangularPart.nr_rectangular_parts
        self.y_min=y_min
        self.y_max=y_max
        self.b=b

    def get_area(self):
        return self.b*(self.y_max-self.y_min)

    def get_centroid(self):
        return 0.5*(self.y_min+self.y_max)

    def get_static_moment(self):
        return self.get_area()*self.get_centroid()

    def draw(self, ax, filled=False):
        # ys=[self.y_min, self.y_min, self.y_max, self.y_max, self.y_min]
        # xs=[-self.b/2, self.b/2, self.b/2, -self.b/2, -self.b/2]
        # ax.plot(xs, ys, color='k', linewidth=2)

        h=self.y_max-self.y_min
        w=self.b

        if filled is False:
            ax.add_patch(Rectangle((-self.b/2, self.y_min),w, h, edgecolor='k', facecolor='w'))
        else:
            ax.add_patch(Rectangle((-self.b / 2, self.y_min), w, h, edgecolor=None, facecolor='k'))

class Event:
    def __init__(self, section, label, description, def_state_structural_steel, def_state_concrete, plotting_list):
        self.section=section
        self.label=str(label)
        self.description=description
        self.def_state_structural_steel=def_state_structural_steel
        self.def_state_concrete=def_state_concrete
        self.plotting_list=plotting_list
        self.color=np.random.rand(3)

    def draw_event_label(self, ax, graphtype):
        xmin, xmax, ymin, ymax = ax.axis()
        dx = xmax - xmin
        dy = ymax - ymin

        if graphtype == 'moment_curvature':
            x = self.def_state_structural_steel.curvature
            y = self.section.get_resultant_M(y_0=self.section.halfheight)/10**6

        if graphtype == 'partial_shear_diagram':
            x = -self.section.get_resultant_N_concrete()/10**3
            y = self.section.get_resultant_M(y_0=self.section.halfheight)/10**6

        if graphtype == 'moment_axialforce_interaction_diagram':
            x = self.section.get_resultant_N()/10**3
            y = self.section.get_resultant_M(y_0=self.section.halfheight)/10**6

        ax.scatter(x, y, color=self.color, marker='s', s=15, zorder=3)

        ax.plot([x,x+ dx / 60, x+ dx / 30],[y, y - dy / 30, y - dy / 30], color=[0.9,0.9,0.9], linewidth=0.5, zorder=1)

        t = ax.text(x + dx / 30, y - dy / 30, self.label, color='w', fontsize=annotations_fontsize,
                    va='center', ha='center', zorder=4)
        t.set_bbox(dict(facecolor=self.color, alpha=1, edgecolor='w', boxstyle='square, pad=0.3'))


    def plot_def_state(self, directory):
        self.section.set_def_state(ds_concrete=self.def_state_concrete, ds_steel=self.def_state_structural_steel)
        self.section.plot_def_state(directory=directory, filename='deformative_state-'+self.label, color=self.color, title=self.label+') '+self.description)

    def plot_on_graph(self,ax, graphtype):
        self.section.set_def_state(ds_concrete=self.def_state_concrete, ds_steel=self.def_state_structural_steel)
        for pg in self.plotting_list:
            if pg==graphtype:
                self.draw_event_label(ax, graphtype)

class Status:
    def __init__(self):
        self.steel_failure=False
        self.concrete_failure=False
        self.steel_yielded=False
        self.concrete_yielded=False
        self.reinforcement_failure=False
        self.reinforcement_yielded=False


