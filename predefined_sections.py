import materials
import section

def generate_section(id):
    steel = materials.StructuralSteel(f_y=460, E=200000, eps_u=0.05, E_hardening=0, law='ELASTIC-PERFECTLY_PLASTIC')
    concrete = materials.Concrete(fck=35, law='PARABOLA-RECTANGLE')
    reinforcement_steel = materials.ReinforcementSteel(f_y=500, E=200000, eps_u=0.01, E_hardening=0, law='ELASTIC-PERFECTLY_PLASTIC')


    if id==1:
        slab_geometry = section.RectangleGeometry(b=5000, h=120, y_min=0)
        profile_geometry = section.HotRolledSection(designation='HE 900 AA', y_min=slab_geometry.h)
        plate_geometry = section.RectangleGeometry(b=500, h=30, y_min=slab_geometry.h+profile_geometry.h)
        reinforcement_geometry = [section.ReinforcementGeometry(area=201 * 5, y=40)]

        slab = section.FlatPart(name='Slab', geometry_description=slab_geometry, material=concrete)
        profile = section.FlatPart(name='Profile', geometry_description=profile_geometry, material=steel)
        plate = section.FlatPart(name='Bottom plate', geometry_description=plate_geometry, material=steel)
        reinforcement = section.ReinforcementPart(name='Reinforcement', geometry_description=reinforcement_geometry, material=reinforcement_steel)

        parts = [slab, profile, plate]
        return section.Section(name='Section-1', parts=parts)

    if id==2:
        slab_geometry = section.RectangleGeometry(b=5000, h=110, y_min=0)
        profile_geometry = section.SingleTGeometry(t_f=20, t_w=15, b=300, h_singleT=420, y_min=110-30)
        reinforcement_geometry = [section.ReinforcementGeometry(area=201 * 5, y=40)]

        slab = section.FlatPart(name='Slab', geometry_description=slab_geometry, material=concrete)
        profile = section.FlatPart(name='Profile', geometry_description=profile_geometry, material=steel)
        reinforcement = section.ReinforcementPart(name='Reinforcement', geometry_description=reinforcement_geometry, material=reinforcement_steel)

        parts = [slab, profile]
        return section.Section(name='Section-2', parts=parts)

    if id==3:
        slab_geometry = section.RectangleGeometry(b=2500, h=70, y_min=0)
        concrete_web_geometry = section.RectangleGeometry(b=200, h=145, y_min=70)
        profile_geometry = section.SingleTGeometry(t_f=35, t_w=15, b=300, h_singleT=250-70+30, y_min=70-30)
        reinforcement_geometry = [section.ReinforcementGeometry(area=201 * 5, y=40)]

        slab = section.FlatPart(name='Slab', geometry_description=slab_geometry, material=concrete)
        concrete_web = section.FlatPart(name='Web', geometry_description=concrete_web_geometry, material=concrete)
        profile = section.FlatPart(name='Profile', geometry_description=profile_geometry, material=steel)
        reinforcement = section.ReinforcementPart(name='Reinforcement', geometry_description=reinforcement_geometry, material=reinforcement_steel)

        parts = [slab, concrete_web, profile]
        return section.Section(name='Section-2', parts=parts)