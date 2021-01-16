import numpy as np
import gmsh


def build_model(model, geometry_parameters, sizes):
    '''GMSH model of Turek benchmark'''
    # Box
    X, Y, H, L = (geometry_parameters[k] for k in ('X', 'Y', 'H', 'L'))
    assert all(x > 0 for x in (X, Y, H, L))

    box_size = sizes.get('box', H/20.)
    
    factory = model.occ
    ll = factory.addPoint(-X, -Y, 0)#, box_size)
    lr = factory.addPoint(-X+L, -Y, 0)#, box_size)
    ur = factory.addPoint(-X+L, -Y+H, 0)#, box_size)
    ul = factory.addPoint(-X, -Y+H, 0)#, box_size)

    lines = [(ll, lr), (lr, ur), (ur, ul), (ul, ll)]
    box_lines = [factory.addLine(*p) for p in lines]
    named_lines = dict(zip(('F_down', 'F_right', 'F_top', 'F_left'), box_lines))

    # Circle
    r, h, l = (geometry_parameters[k] for k in ('r', 'h', 'l'))
    assert all(x > 0 for x in (r, h, l))
    #
    #      /
    # ----/
    center = factory.addPoint(0, 0, 0)#, 1)
    sym = factory.addPoint(-r, 0, 0)#, 1)#sizes.get('circle', r/40.))

    flap_size_front = 1#sizes.get('flap_front', h/5.)

    dx = np.sqrt(r**2 - (0.5*h)**2)
    top = factory.addPoint(dx, 0.5*h, 0)#, flap_size_front)
    bottom = factory.addPoint(dx, -0.5*h, 0)#, flap_size_front)

    circle_lines = [factory.addCircleArc(top, center, sym),
                    factory.addCircleArc(sym, center, bottom)]
    named_lines.update(dict(zip(('F_circle_top', 'F_circle_bottom'), circle_lines)))

    # Flap
    flap_size_back = sizes.get('flap_front', h/5.)
    ftop = factory.addPoint(dx+l, 0.5*h, 0)#, flap_size_back)
    fbottom = factory.addPoint(dx+l, -0.5*h, 0)#, flap_size_back)

    lines = [(bottom, fbottom), (fbottom, ftop), (ftop, top), (top, bottom)]
    flap_lines = [factory.addLine(*p) for p in lines]                   
    named_lines.update(dict(zip(('I_bottom', 'I_right', 'I_top', 'S_left'), flap_lines)))

    box_loop = factory.addCurveLoop(box_lines)
    hole = factory.addCurveLoop(circle_lines + flap_lines[:-1])
    fluid = factory.addPlaneSurface([box_loop, hole])

    solid_loop = factory.addCurveLoop(flap_lines)
    solid = factory.addPlaneSurface([solid_loop])

    factory.synchronize()

    tags = {'cell': {'F': 1, 'S': 2},
            'facet': {}}
    # Physical tags
    model.addPhysicalGroup(2, [fluid], 1)
    model.addPhysicalGroup(2, [solid], 2)

    for name in named_lines:
        if 'circle' not in name:
            tag = named_lines[name]
            model.addPhysicalGroup(1, [tag], tag)
            tags['facet'][name] = tag
    # Two pieces but one physical
    tag = named_lines['F_circle_top']
    model.addPhysicalGroup(1, [tag, named_lines['F_circle_bottom']], tag)
    tags['facet']['circle'] = tag

    # return model, tags
    factory.synchronize()

    return model, tags
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from sleep.mesh import mesh_model2d, load_mesh2d
    import dolfin as df
    import sys

    if '-format' not in sys.argv:
        sys.argv.extend(['-format', 'msh2'])
    
    gmsh.initialize(sys.argv)

    model = gmsh.model

    geometry_params = {'r': 1.,
                       'h': 0.2,
                       'l': 5.,
                       'X': 2.,
                       'Y': 2.,
                       'H': 5.,
                       'L': 10.}
    model, tags = build_model(model, geometry_params, {})

    l, h = geometry_params['l'], geometry_params['h']
    x = np.sqrt(geometry_params['r']**2 - (0.5*h)**2)
    
    field = model.mesh.field
    field.add('Box', 1)
    field.setNumber(1, 'XMin', 0)
    field.setNumber(1, 'XMax', x+l)
    field.setNumber(1, 'YMin', -0.5*h)
    field.setNumber(1, 'YMax', 0.5*h)
    field.setNumber(1, 'VIn', h/5.)
    field.setNumber(1, 'VOut', geometry_params['H']/20.)
    
    field.setAsBackgroundMesh(1)

    model.occ.synchronize()

    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
    
    h5_filename = './test/turek_domain.h5'
    mesh_model2d(model, tags, h5_filename)

    mesh, markers, lookup = load_mesh2d(h5_filename)
    print lookup
    gmsh.finalize()
    
    df.File('./test/turek_cells.pvd') << markers[0]
    df.File('./test/turek_facets.pvd') << markers[1]
