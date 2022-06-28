from sleep.fbb_DD.plane_biot import pcws_constant
import dolfin as df


def test_scalar():
    mesh = df.UnitSquareMesh(4, 4)
    cell_f = df.MeshFunction('size_t', mesh, 2, 0)
    df.CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS').mark(cell_f, 1)

    foo, bar = df.Constant(1), df.Constant(2)
    cases = {0: foo, 1: bar}

    f = pcws_constant(cell_f, cases)
    dx = df.Measure('dx', domain=mesh, subdomain_data=cell_f)

    L = df.inner(f-foo, f-foo)*dx(0) + df.inner(f-bar, f-bar)*dx(1)
    e = df.assemble(L)

    assert df.sqrt(abs(e)) < 1E-14


def test_vector():
    mesh = df.UnitSquareMesh(4, 4)
    cell_f = df.MeshFunction('size_t', mesh, 2, 0)
    df.CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS').mark(cell_f, 1)

    foo, bar = df.Constant((1, 1)), df.Constant((2, 2))
    cases = {0: foo, 1: bar}

    f = pcws_constant(cell_f, cases)
    dx = df.Measure('dx', domain=mesh, subdomain_data=cell_f)

    L = df.inner(f-foo, f-foo)*dx(0) + df.inner(f-bar, f-bar)*dx(1)
    e = df.assemble(L)

    assert df.sqrt(abs(e)) < 1E-14

    
def test_tensor():
    mesh = df.UnitSquareMesh(4, 4)
    cell_f = df.MeshFunction('size_t', mesh, 2, 0)
    df.CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS').mark(cell_f, 1)

    foo, bar = df.Constant(((1, 1), (0, 1))), df.Constant(((2, 2), (3, 4)))
    cases = {0: foo, 1: bar}

    f = pcws_constant(cell_f, cases)
    dx = df.Measure('dx', domain=mesh, subdomain_data=cell_f)

    L = df.inner(f-foo, f-foo)*dx(0) + df.inner(f-bar, f-bar)*dx(1)
    e = df.assemble(L)

    # Axisym?
    df.as_matrix(((foo[0, 0], foo[0, 1], df.Constant(0)),
                  (foo[1, 0], foo[1, 1], df.Constant(0)),
                  (df.Constant(0), df.Constant(0), df.Constant(1))))

    assert df.sqrt(abs(e)) < 1E-14


def test_notags():
    mesh = df.UnitSquareMesh(4, 4)
    cell_f = df.MeshFunction('size_t', mesh, 2, 0)
    df.CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS').mark(cell_f, 1)

    foo, bar = df.Constant(1), df.Constant(2)
    cases = {0: foo}

    try:
        f = pcws_constant(cell_f, cases)
    except AssertionError:
        assert True
