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

def check_markers(cell_f, facet_f, lookup, geometry_parameters):
    '''Things are where we expect them'''
    # Box
    X, Y, H, L = (geometry_parameters[k] for k in ('X', 'Y', 'H', 'L'))
    # Circle and flap
    r, h, l = (geometry_parameters[k] for k in ('r', 'h', 'l'))

    dx = np.sqrt(r**2 - (0.5*h)**2)
    # Check boundary marking first
    positions = {'F_left': np.array([-X, -Y+0.5*H]),
                 'F_down': np.array([-X+0.5*L, -Y]),
                 'F_right': np.array([-X+L, -Y+0.5*H]),
                 'F_top': np.array([-X+0.5*L, -Y+H]),
                 'S_left': np.array([dx, 0]),
                 'I_top': np.array([dx+0.5*l, 0.5*h]),
                 'I_right': np.array([dx+l, 0]),
                 'I_bottom': np.array([dx+0.5*l, -0.5*h])}

    mesh = facet_f.mesh()
    x = mesh.coordinates()
    
    mesh.init(1, 0)
    e2v = mesh.topology()(1, 0)
    
    tagged_vertices = lambda tag: np.unique(np.hstack(map(e2v, np.where(facet_f.array() == tag)[0])))
    center = lambda coord: 0.5*(np.max(coord, axis=0) + np.min(coord, axis=0))

    for tag in positions:
        target = positions[tag]
        vertices = x[tagged_vertices(lookup['facet'][tag])] 
        assert np.linalg.norm(target - center(vertices)) < 1E-13
    # Finally make sure that cirle points are radiues away from center
    vertices = x[tagged_vertices(lookup['facet']['circle'])] 
    assert np.all(abs(np.linalg.norm(vertices, 2, axis=1)-r) < 1E-13)
    
    # Now subdomain marking
    positions = {'S': np.array([dx+0.5*l, 0])}
    
    mesh.init(1, 0)
    c2v = mesh.topology()(2, 0)
    
    tagged_vertices = lambda tag: np.unique(np.hstack(map(c2v, np.where(cell_f.array() == tag)[0])))
    center = lambda coord: np.array([0.5*(np.min(coord[:, 0]) + np.max(coord[:, 0])),
                                     0.5*(np.min(coord[:, 1]) + np.max(coord[:, 1]))])

    unique_tags = set(cell_f.array())
    for tag in positions:
        target = positions[tag]
        unique_tags.remove(lookup['cell'][tag])
        vertices = x[tagged_vertices(lookup['cell'][tag])] 
        assert np.linalg.norm(target - center(vertices)) < 1E-13

    fluid_tag, = unique_tags
    assert lookup['cell']['F'] == fluid_tag
                 
    return True
    
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
    gmsh.finalize()
    
    mesh, markers, lookup = load_mesh2d(h5_filename)
    cell_f, facet_f = markers
    
    check_markers(cell_f, facet_f, lookup, geometry_params)    
    
    df.File('./test/turek_cells.pvd') << markers[0]
    df.File('./test/turek_facets.pvd') << markers[1]
