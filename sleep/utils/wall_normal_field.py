import dolfin as df


class WallNormalField(df.UserExpression):
    '''Wall normal field of 2D mesh (well defined only on exterior facets)'''
    # NOTE: this will point outwards!!!
    def __init__(self, mesh, magnitude=df.Constant(1), **kwargs):
        super().__init__(kwargs)
        self.mesh = mesh
        self.magnitude = magnitude

    def eval_cell(self, value, x, ufc_cell):
        cell = df.Cell(self.mesh, ufc_cell.index)
        self.mesh.init(1)
        self.mesh.init(2, 1)

        x = df.Point(x)
        mp = cell.midpoint()
        # I assume that we will use RT1 in eval - then the dof sits on
        # one of the midpoints
        facet = min([df.Facet(mesh, idx) for idx in cell.entities(1)],
                    key=lambda f: f.midpoint().distance(x))
        assert facet.midpoint().distance(x) < 1E-13, (facet.midpoint().distance(x), x.array())

        n = facet.normal()
        # Orient to point outside of the cell
        if n.dot(facet.midpoint() - mp) < 0:
            n = -n.array()[:2]
        else:
            n = n.array()[:2]
            
        # Scale
        if isinstance(self.magnitude, (df.Constant, int, float)):
            mag = self.magnitude(0)
        else:
            mag = self.magnitude(facet.midpoint().array())
        value[:] = mag*n
        
    def value_shape(self):
        return (2, )

# --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np
    
    mesh = df.UnitSquareMesh(32, 32)

    f = WallNormalField(mesh, degree=0)

    V = df.FunctionSpace(mesh, 'RT', 1)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)

    K = df.FacetArea(mesh)
    # The degree of freedom RT1 is int_edge u.n = u(mid).n(mid)*edge_len
    true = df.assemble(K*df.dot(n, v)*df.ds)
    # So these are point evals On the boundary
    true = true.get_local()
    
    ans = df.interpolate(f, V).vector()
    ans = ans.get_local()
    # We are only interested in comparison on exterior facets
    idx = np.where(np.abs(true) > 0)
    
    assert np.linalg.norm(ans[idx] - true[idx]) < 1E-13
