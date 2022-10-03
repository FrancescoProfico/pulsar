import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font={'family': 'Arial', 'weight': 'normal', 'size': 8}
matplotlib.rc('font',**font)

font_axttitle={'family': 'Arial', 'weight': 'bold', 'size': 8}
font_axlabel={'family': 'Arial', 'weight': 'bold', 'size': 7}

annotations_fontsize=6

def draw_baseline(ax, section):
    baseaxis = np.array([[0, section.get_y_min()], [0, section.get_y_max()]])
    ax.plot(baseaxis[:, 0], baseaxis[:, 1], color='k', linewidth=3, zorder=3)

def deformative_state_chart_setup(ax, section):
    ax.set_xlabel('Strain - '+r'$\epsilon\:[-]$', **font_axlabel)
    ax.set_ylabel('Coordinate - ' + r'$y\:[mm]$', **font_axlabel)
    ax.set_title('Deformative state', **font_axttitle)
    ax.grid(color=[0.95,0.95,0.95])
    ax.invert_yaxis()
    ax.axis('off')

    draw_baseline(ax, section)

def material_law_chart_setup(ax, material):
    ax.set_xlabel('Strain - ' + r'$\epsilon\:[-]$', **font_axlabel)
    ax.set_ylabel('Stress - ' + r'$\sigma\:[MPa]$', **font_axlabel)
    ax.set_title(material.name+' - material constitutive law', **font_axttitle)
    ax.grid(color=[0.95, 0.95, 0.95])

def stress_state_chart_setup(ax, section):
    ax.set_xlabel('Stress - '+r'$\sigma\:[MPa]$', **font_axlabel)
    ax.set_ylabel('Coordinate - ' + r'$y\:[mm]$', **font_axlabel)
    ax.set_title('Stress state', **font_axttitle)
    ax.grid(color=[0.95,0.95,0.95])
    ax.invert_yaxis()
    ax.axis('off')

    draw_baseline(ax, section)

def section_chart_setup(ax):
    ax.set_xlabel(r'[mm]', **font_axlabel)
    ax.set_ylabel('Coordinate - ' + r'$y\:[mm]$', **font_axlabel)
    ax.set_title('Section', **font_axttitle)
    ax.grid(color=[0.95, 0.95, 0.95])
    ax.invert_yaxis()
    ax.set_xlim(-500,500)
    ax.set_aspect('equal')
    ax.axis('off')

def partial_resultants_chart_setup(ax, section):
    ax.set_xlabel('', **font_axlabel)
    ax.set_ylabel('Coordinate - ' + r'$y\:[mm]$', **font_axlabel)
    ax.set_title('Partial resultants', **font_axttitle)
    ax.grid(color=[0.95, 0.95, 0.95])
    draw_baseline(ax, section)
    ax.invert_yaxis()
    ax.axis('off')

    ax.set_xlim(-0.5, 4)

def global_resultants_chart_setup(ax, section):
    ax.set_xlabel('', **font_axlabel)
    ax.set_ylabel('Coordinate - ' + r'$y\:[mm]$', **font_axlabel)
    ax.set_title('Global resultants', **font_axttitle)
    ax.grid(color=[0.95, 0.95, 0.95])
    draw_baseline(ax, section)
    ax.invert_yaxis()
    ax.axis('off')

    ax.set_xlim(-0.5, 4)

def moment_curvature_chart_setup(ax):
    ax.set_xlabel('Curvature - '+r'$(1/r)\:[1/mm]$', **font_axlabel)
    ax.set_ylabel('Moment - ' + r'$M\:[kNm]$', **font_axlabel)
    ax.set_title('Moment curvature diagram', **font_axttitle)
    ax.grid(color=[0.95, 0.95, 0.95])

def partial_shear_diagram_chart_setup(ax):
    ax.set_xlabel('Compression on concrete part - '+r'$N_c\:[kN]$', **font_axlabel)
    ax.set_ylabel('Moment - ' + r'$M\:[kNm]$', **font_axlabel)
    ax.set_title('Partial Shear Diagram', **font_axttitle)
    ax.plot([0],[0])
    ax.grid(color=[0.95, 0.95, 0.95])

def moment_axialforce_interaction_chart_setup(ax):
    ax.invert_xaxis()
    ax.set_xlabel('Axial force - '+r'$N\:[kN]$', **font_axlabel)
    ax.set_ylabel('Moment - '+r'$M\:[kNm]$', **font_axlabel)
    ax.set_title('Moment-axial force interaction diagram', **font_axttitle)
    ax.grid(color=[0.95, 0.95, 0.95])

def neutralaxis_axialforce_chart_setup(ax, section):
    ax.set_xlabel('Neutral axis position - ' + r'$y_{pl}\:[mm]$', **font_axlabel)
    ax.set_ylabel('Axial force - ' + r'$N\:[kN]$', **font_axlabel)
    ax.set_title('Neutral axis-axial force diagram', **font_axttitle)
    ax.grid(color=[0.95, 0.95, 0.95])
    ax.set_xlim(section.y_min-100, section.y_max+100)

def neutralaxis_ultcurvature_chart_setup(ax, section):
    ax.set_xlabel('Neutral axis position - ' + r'$y_{pl}\:[mm]$', **font_axlabel)
    ax.set_ylabel('Ultimate curvature - ' + r'$(1/r)_u\:[1/mm]$',**font_axlabel)
    ax.set_title('Neutral axis-ultimate curvature diagram', **font_axttitle)
    ax.grid(color=[0.95, 0.95, 0.95])
    ax.set_xlim(section.y_min - 100, section.y_max + 100)



