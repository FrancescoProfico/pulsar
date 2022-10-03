import predefined_sections
import numpy as np
import section

work_directory=r'./pdfs/'

s=predefined_sections.generate_section(1)

s.plot_section(work_directory)
s.uls_fsi(axial_force=0)
s.plot_partial_shear_diagram(work_directory)
s.plot_moment_axialforce_interaction_diagram(work_directory)
s.plot_moment_curvature_diagram(work_directory)
s.plot_events_def_state(work_directory)

