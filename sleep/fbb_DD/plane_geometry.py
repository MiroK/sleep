from gmshnics import msh_gmsh_model, mesh_from_gmsh
from itertools import combinations
import numpy as np
import gmsh


def add_circle(cx, cy, r, factory):
    '''4 pieces of the boundary'''
    center = factory.addPoint(cx, cy, 0, r)
    points = [factory.addPoint(cx+r, cy, 0, r),
              factory.addPoint(cx, cy+r, 0, r),
              factory.addPoint(cx-r, cy, 0, r),
              factory.addPoint(cx, cy-r, 0, r)]
    
    lines = [factory.addCircleArc(points[i], center, points[(i+1)%4])
             for i in range(4)]

    return lines
    

def plane_geometry(ll, ur, holes, scale, *args):
    '''
    Slice [ll[0], ur[0]] x [ll[1], ur[1]] with holes where hole is defined
    by (center-x, center-y, inner radius, outer radius). We return a    
    '''
    ll, ur = map(np.array, (ll, ur))
    dx, dy = ur - ll
    assert dx > 0 and dy > 0
    
    assert all(r1 < r2 for (cx, cy, r1, r2) in holes)
    
    # All circles are inside
    x0, y0 = ll
    x1, y1 = ur
    assert all((x0 < cx - r2 < cx + r2 < x1) and (y0 < cy - r2 < cy + r2 < y1)
               for (cx, cy, r1, r2) in holes)

    # The circles do not overlap
    assert all(np.linalg.norm([cy_ - cy, cx_ - cx], 2) > r2 + r2_
               for (cx, cy, r1, r2), (cx_, cy_, r1_, r2_) in combinations(holes, 2))

    gmsh.initialize(['', '-clscale', str(scale)])

    model = gmsh.model
    factory = model.occ
    # Let's make outer boundary
    # DC
    # AB
    A = factory.addPoint(x0, y0, 0)
    B = factory.addPoint(x1, y0, 0)
    C = factory.addPoint(x1, y1, 0)
    D = factory.addPoint(x0, y1, 0)

    boundaries = [factory.addLine(*l) for l in ((A, B), (B, C), (C, D), (D, A))]
    boundary_loop = factory.addCurveLoop(boundaries)

    circles, surfs, outer_loops = [], [], []
    for cx, cy, r1, r2 in holes:
        inner = add_circle(cx, cy, r1, factory)
        inner_loop = factory.addCurveLoop(inner)
        
        outer = add_circle(cx, cy, r2, factory)
        outer_loop = factory.addCurveLoop(outer)
        outer_loops.append(outer_loop)
        
        circles.append((inner, outer))
        
        surfs.append(factory.addPlaneSurface([inner_loop, outer_loop]))
    outer_surface = factory.addPlaneSurface([boundary_loop] + outer_loops)

    factory.synchronize()    
    # Finally for tagging;
    # We tag facets such that
    #                     4
    #                   1   2
    #                     3
    # Then each circle gets to name its true boundary (and finally we label)
    # then interface with the tissue. With volumes the tissue gets tag 1
    # and then volume id of the circle is >= 5 (so that it matches its boundary
    # tag)
    
    model.addPhysicalGroup(1, [boundaries[3]], 1)
    model.addPhysicalGroup(1, [boundaries[1]], 2)
    model.addPhysicalGroup(1, [boundaries[0]], 3)
    model.addPhysicalGroup(1, [boundaries[2]], 4)
    
    for tag, circle in enumerate(circles, 5):
        model.addPhysicalGroup(1, circle[0], tag)

    for tag, circle in enumerate(circles, tag+1):
        model.addPhysicalGroup(1, circle[1], tag)

    model.addPhysicalGroup(2, [outer_surface], 1)
    for tag, surf in enumerate(surfs, 5):
        model.addPhysicalGroup(2, [surf], tag)
        
    factory.synchronize()

    nodes, topologies = msh_gmsh_model(model, 2)
    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return entity_functions

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import File

    ll = (0, 0)
    ur = (2, 2)
    holes = [(0.35, 0.35, 0.1, 0.2),
             (1.65, 1.65, 0.1, 0.2)]
    
    entity_fs = plane_geometry(ll, ur, holes, scale=0.4)

    File('surfs.pvd') << entity_fs[2]
    File('facets.pvd') << entity_fs[1]    
