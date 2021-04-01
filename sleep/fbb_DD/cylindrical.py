# Cylindrical and axisym operators
# NOTE: axis ordering e_z, e_r, e_theta is assumed
import dolfin as df 


def Grad(u):
    '''Gradient in cylinderical (z, r, theta) coords'''
    z, r, th = df.SpatialCoordinate(u.ufl_domain().ufl_cargo())
    # Scalar
    if u.ufl_shape == ():
        return df.as_vector((u.dx(0), u.dx(1), (1/r)*u.dx(2)))

    # Ignore tensors for now
    assert len(u.ufl_shape) == 1
    # z r theta 
    uz, ur, uth = u[0], u[1], u[2]
    return df.as_matrix(((uz.dx(0),  uz.dx(1),  (1/r)*uz.dx(2)),
                         (ur.dx(0),  ur.dx(1),  (1/r)*ur.dx(2) - uth/r),
                         (uth.dx(0), uth.dx(1), (1/r)*uth.dx(2) + ur/r)))

def Div(u):
    '''Divergence in cylindrical coordinates'''
    if len(u.ufl_shape) == 1:
        return df.tr(Grad(u))
    assert len(u.ufl_shape) == 2
    return df.as_vector(tuple(Div(u[i, :]) for i in range(u.ufl_shape[0])))


def GradAxisym(u):
    '''Gradient in cylinderical axisymmetric (z, r) coords'''
    z, r = df.SpatialCoordinate(u.ufl_domain().ufl_cargo())
    # Scalar
    if u.ufl_shape == ():
        return df.as_vector((u.dx(0), u.dx(1), df.Constant(0)*u))

    # Ignore tensors for now
    assert len(u.ufl_shape) == 1
    # z r theta 
    uz, ur = u[0], u[1]
    return df.as_matrix(((uz.dx(0),  uz.dx(1),  df.Constant(0)*uz),
                         (ur.dx(0),  ur.dx(1),  df.Constant(0)*ur),
                         (df.Constant(0)*uz, df.Constant(0)*ur, ur/r)))
    

def DivAxisym(u):
    '''Divergence in cylindrical axisymmetric coordinates'''
    if len(u.ufl_shape) == 1:
        return df.tr(GradAxisym(u))
    assert len(u.ufl_shape) == 2
    return df.as_vector(tuple(DivAxisym(u[i, :]) for i in range(u.ufl_shape[0])))
