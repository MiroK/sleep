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
    
    named_lines = dict(zip(('F_left', 'F_down', 'F_right'), fluid_lines))
    named_lines['I'] = interface[0]
    
    fluid_loop = factory.addCurveLoop(fluid_lines + interface)
    fluid = factory.addPlaneSurface([fluid_loop])

    # Solid is top
    lines = [(mr, ur), (ur, ul), (ul, ml)]    
    solid_lines = [factory.addLine(*p) for p in lines]
    named_lines.update(dict(zip(('S_right', 'S_top', 'S_left'), solid_lines)))
    
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
        tags['facet'][name] = tag

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

    # Origin, width, inside, outside sizes
    sizes = {'F': (0, 0.1, 0.05, 0.2),
             'I': (0+geometry_parameters['YF']-0.1, 0.2, 0.01, 0.2),
             'S': (0+geometry_parameters['YF']+geometry_parameters['YS']-0.1, 0.1, 0.2, 0.3)}

    field = model.mesh.field
    fid = 1
    boxes = []
    for (y, w, Vin, Vout) in sizes.values():
         field.add('Box', fid)
         field.setNumber(fid, 'XMin', 0)
         field.setNumber(fid, 'XMax', 0+geometry_parameters['X'])
         field.setNumber(fid, 'YMin', y)
         field.setNumber(fid, 'YMax', y+w)
         field.setNumber(fid, 'VIn', Vin)
         field.setNumber(fid, 'VOut', Vout)

         boxes.append(fid)
         fid += 1
    # Combine
    field.add('Min', fid)
    field.setNumbers(fid, 'FieldsList', boxes)    
    field.setAsBackgroundMesh(fid)

    model.occ.synchronize()
    
    gmsh.fltk.initialize()
    gmsh.fltk.run()
    
    h5_filename = './test/sleep_domain.h5'
    mesh_model2d(model, tags, h5_filename)

    mesh, markers, lookup = load_mesh2d(h5_filename)
    
    gmsh.finalize()
    
    df.File('./test/sleep_cells.pvd') << markers[0]
    df.File('./test/sleep_facets.pvd') << markers[1]
