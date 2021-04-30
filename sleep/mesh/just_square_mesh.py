# |---------------------|
# |                     |
# Y                     |
# |                     |
# |---------X-----------|
# origin
import numpy as np
import gmsh


def build_model(model, geometry_parameters):
    '''GMSH model for sleep with pulsating left wall'''

    X, Y = (geometry_parameters[k] for k in ('X', 'Y'))
    assert X > 0 and Y > 0
    
    factory = model.occ
    ll = factory.addPoint(0, 0, 0)
    lr = factory.addPoint(X, 0, 0)
    ur = factory.addPoint(X, Y, 0)
    ul = factory.addPoint(0, Y, 0)


    lines = [(ll, lr), (lr, ur), (ur, ul), (ul, ll)]
    
    fluid_lines = [factory.addLine(*p) for p in lines]
    named_lines = dict(zip(('bottom', 'right', 'top', 'left'), fluid_lines))
    
    fluid_loop = factory.addCurveLoop(fluid_lines)
    fluid = factory.addPlaneSurface([fluid_loop])

    factory.synchronize()
    
    tags = {'cell': {'F': 1},
            'facet': {}}
    # Physical tags
    model.addPhysicalGroup(2, [fluid], 1)

    for name in named_lines:
        tag = named_lines[name]
        model.addPhysicalGroup(1, [tag], tag)
        tags['facet'][name] = tag

    factory.synchronize()

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

    geometry_parameters = {'X': 1, 'Y': 1}
    model, tags = build_model(model, geometry_parameters)

    # Origin, width, inside, outside sizes
    fid = 1

    field = model.mesh.field    
    # Refine close to left
    field.add('Box', fid)
    field.setNumber(fid, 'XMin', 0)
    field.setNumber(fid, 'XMax', 0.1*geometry_parameters['X'])
    field.setNumber(fid, 'YMin', 0)
    field.setNumber(fid, 'YMax', geometry_parameters['Y'])
    field.setNumber(fid, 'VIn', 0.05)
    field.setNumber(fid, 'VOut', 0.2)

    field.setAsBackgroundMesh(fid)

    model.occ.synchronize()
    
    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
    
    h5_filename = './test/square_domain.h5'
    mesh_model2d(model, tags, h5_filename)

    mesh, markers, lookup = load_mesh2d(h5_filename)
    cell_f, facet_f = markers
    
    gmsh.finalize()
    
    df.File('./test/square_cells.pvd') << cell_f
    df.File('./test/square_facets.pvd') << facet_f
