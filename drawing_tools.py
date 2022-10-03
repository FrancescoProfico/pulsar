import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

from plot_setup import annotations_fontsize

def draw_axis(ax):
    xmin, xmax, ymin, ymax = ax.axis()
    xax=np.array([[xmin,0],[xmax,0]])
    yax=np.array([[0,ymin],[0,ymax]])

    ax.plot(xax[:,0], xax[:,1], color='k', linewidth=0.3)
    ax.plot(yax[:, 0], yax[:, 1], color='k', linewidth=.3)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

def draw_partial_resultant_arrow(ax, y, value, string):
    length=0.2
    head_length=0.2
    base_offset=0.2
    dy=0
    if value>=0:
        offset = base_offset
        x=offset
        dx=length
    else:
        offset = base_offset+head_length
        x=offset+length
        dx=-length

    ax.arrow(x, y, dx, dy, width=8, head_width=20, head_length=head_length, zorder=1, fc='k', ec=None)
    t = ax.text(base_offset+head_length+length+0.5*base_offset, y, string+': '+'N='+str(round(abs(value)/1000,1))+' kN; y='+str(round(y,1))+' mm', color='k', fontsize=annotations_fontsize, va='center')
    t.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='square, pad=0.1'))

def draw_global_resultant_arrow(ax, y, value):
    length=0.2
    head_length=0.2
    base_offset=0.5
    dy=0
    if value>=0:
        offset = base_offset
        x=offset
        dx=length
    else:
        offset = base_offset+head_length
        x=offset+length
        dx=-length

    ax.arrow(x, y, dx, dy, width=8, head_width=20, head_length=head_length, zorder=1, fc='k', ec=None)
    t = ax.text(base_offset+head_length+length+0.5*base_offset, y, 'N='+str(round(abs(value)/1000,1))+' kN', color='k', fontsize=annotations_fontsize, va='center')
    t.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='square, pad=0.1'))

def draw_moment_resultant_arrow(ax, y, value):
    base_offset = 0.2

    ymin, ymax=ax.get_ylim()
    deltay=abs(ymax-ymin)

    draw_arc_arrow(ax, base_offset, y, deltay/32, deltay/8)

    t = ax.text(3*base_offset, y-deltay/8, 'M=' + str(round(abs(value) / 1000000, 1)) + ' kNm', color='k', fontsize=annotations_fontsize, va='center')
    t.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='square, pad=0.1'))

def draw_arc_arrow(ax, x, y, dx, dy):

    xmin, xmax, ymin, ymax = ax.axis()
    r=abs((xmax-xmin)/(ymax-ymin))
    head_length = 20

    discr=15
    l=(dx**2+dy**2)**0.5
    phi=np.arctan(dy/dx)
    tot_angle=2*(np.pi-2*phi)
    rad=l/2/np.cos(phi)
    dangle=tot_angle/(discr-1)
    angles=[-tot_angle/2+dangle*i for i in range(discr)]
    xx=[r*(rad*np.cos(a)-rad+dx)+x for a in angles]
    yy=[y+rad*np.sin(a) for a in angles]

    ax.plot(xx[:-1], yy[:-1], color='k', linewidth=2.5)

    # angle=30*np.pi/180
    # c=np.cos(angle)
    # s=np.sin(angle)
    # arrowhead=np.array([[-20,0,20],[0,60,0]])
    # rot=np.array([[c,-s],[s,c]])
    # arrowhead=np.dot(rot, arrowhead)
    # dist=np.array([[r,0],[0,1]])
    # arrowhead = np.dot(dist, arrowhead)
    # translation_x=0.2
    # translation_y= 50
    # tx=[translation_x for i in range(3)]
    # ty=[translation_y for i in range(3)]
    # transl=[tx,ty]
    # arrowhead=np.add(arrowhead, transl)
    # ax.plot(arrowhead[0,:], arrowhead[1,:], color='k', linewidth=2.5)

def write_measure(ax, x, y, ha, label, amount, meas_unit):
    string=label+'='+str(round(amount,1))+' '+meas_unit
    t = ax.text(x, y, string, color='k', fontsize=annotations_fontsize, va='center', ha=ha)
    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', boxstyle='square, pad=0.0'))
