# |---------------------|
# YS                    |
# |---------------------|
# YF                    |
# |---------X-----------|
# origin
import numpy as np
import gmsh


def build_model(model, geometry_parameters):
    '''GMSH model of sleep 2d study'''

    X, YF, YS = (geometry_parameters[k] for k in ('X', 'YF', 'YS'))
    assert X > 0 and YF > 0 and YS > 0
    
    factory = model.occ
    ll = factory.addPoint(0, 0, 0)
    lr = factory.addPoint(X, 0, 0)
    mr = factory.addPoint(X, YF, 0)
    ur = factory.addPoint(X, YF+YS, 0)
    ul = factory.addPoint(0, YF+YS, 0)
    ml = factory.addPoint(0, YF, 0)

    # Bottom part is fluid
    lines = [(ml, ll), (ll, lr), (lr, mr), (mr, ml)]
    
    fluid_lines = [factory.addLine(*p) for p in lines]
    interface = [fluid_lines.pop()]
    
    named_lines = dict(zip(('F_down', 'F_right', 'F_left'), fluid_lines))
    named_lines['I'] = interface[0]
    
    fluid_loop = factory.addCurveLoop(fluid_lines + interface)
    fluid = factory.addPlaneSurface([fluid_loop])

    # Solid is top
    lines = [(mr, ur), (ur, ul), (ul, ml)]    
    solid_lines = [factory.addLine(*p) for p in lines]
    named_lines.update(dict(zip(('S_left', 'S_top', 'S_right'), solid_lines)))
    
    solid_loop = factory.addCurveLoop(interface + solid_lines)
    solid = factory.addPlaneSurface([solid_loop])

    factory.synchronize()

    tags = {'cell': {'F': 1, 'S': 2},
            'facet': {}}
    # Physical tags
    model.addPhysicalGroup(2, [fluid], 1)
    model.addPhysicalGroup(2, [solid], 2)

    for name in named_lines:
        tag = named_lines[name]
        model.addPhysicalGroup(1, [tag], tag)
        tags['facet'][tag] = tag

    return model, tags

# --------------------------------------------------------------------

if __name__ == '__main__':
    from sleep.mesh import mesh_model2d, load_mesh2d, set_mesh_size
    import dolfin as df
    import sys

    if '-format' not in sys.argv:
        sys.argv.extend(['-format', 'msh2'])
    
    gmsh.initialize(sys.argv)

    model = gmsh.model

    geometry_parameters = {'X': 2, 'YF': 0.5, 'YS': 0.25}
    model, tags = build_model(model, geometry_parameters)

    mesh_sizes = {'DistMin': 0.1, 'DistMax': 1, 'LcMin': 0.1, 'LcMax': 0.2}    
    set_mesh_size(model, tags, mesh_sizes)
    
    h5_filename = './test/sleep_domain.h5'
    mesh_model2d(model, tags, h5_filename)

    mesh, markers, lookup = load_mesh2d(h5_filename)
    
    gmsh.finalize()
    
    df.File('./test/sleep_cells.pvd') << markers[0]
    df.File('./test/sleep_facets.pvd') << markers[1]
