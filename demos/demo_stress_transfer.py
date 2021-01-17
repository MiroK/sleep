# A use case: Let there be a deformation on solid domain. Now in the
# fluid solver we'd like to evaluate it on the interface. What could
# work is
#
#     us -> take is trace Tus -> extend to fluid -> compute the stress there
#
# You'll that this does not work and the work around is to take trace of
# stresses and extend those
from sleep.utils import EmbeddedMesh, trace_matrix
from dolfin import *

fs_domain = UnitSquareMesh(128, 128)
cell_f = MeshFunction('size_t', fs_domain, 2, 1)
CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS').mark(cell_f, 2)

dx_fs = Measure('dx', domain=fs_domain, subdomain_data=cell_f)

fluid = EmbeddedMesh(cell_f, 1)
solid = EmbeddedMesh(cell_f, 2)

# We take fluid domain and master and define interaface wrt to it
fluid_facets = MeshFunction('size_t', fluid, 1, 0)
CompiledSubDomain('near(x[0], 0.5)').mark(fluid_facets, 1)
# Corresponding suraface integral
dI_f = Measure('ds', domain=fluid, subdomain_data=fluid_facets, subdomain_id=1)
# Fluid is master
interface = EmbeddedMesh(fluid_facets, 1)

solid_facets = MeshFunction('size_t', solid, 1, 0)
CompiledSubDomain('near(x[0], 0.5)').mark(solid_facets, 2)
# Corresponding suraface integral
dI_s = Measure('ds', domain=solid, subdomain_data=solid_facets, subdomain_id=2)

def get_u(mesh):
    '''Ground truth'''
    x, y = SpatialCoordinate(mesh)
    return as_vector((x**2+y**2, x-2*y**2))

uF0 = get_u(fluid)

F = VectorFunctionSpace(fluid, 'CG', 2)
S = VectorFunctionSpace(solid, 'CG', 2)
uF = project(uF0, F)

TF = VectorFunctionSpace(interface, 'CG', 2)
F2iface = trace_matrix(F, TF, interface)
S2iface = trace_matrix(S, TF, interface)

TuF = Function(TF) # Intermediate step
uS = Function(S)   # Destination

F2iface.mult(uF.vector(), TuF.vector())
S2iface.transpmult(TuF.vector(), uS.vector())

# For comparison
uS0 = get_u(solid)

n = Constant((-1, 0))
# Eval some functions now
es = [inner(uS0 - uS, uS0 - uS)*dI_s,
      inner(dot(grad(uS0), n) - dot(grad(uS), n),
            dot(grad(uS0), n) - dot(grad(uS), n))*dI_s]

for e in es:
    print sqrt(abs(assemble(e)))
# So we see that "sharing" displacement is not enough
# The other option is project the derived quantity, and transform
# its trace
F_grad = TensorFunctionSpace(fluid, 'DG', 1)
stress_f = project(grad(uF), F_grad)

TF_grad = TensorFunctionSpace(interface, 'DG', 1)
Tstress = Function(TF_grad)
trace_matrix(F_grad, TF_grad, interface).mult(stress_f.vector(), Tstress.vector())

S_grad = TensorFunctionSpace(solid, 'DG', 1)
stress_s = Function(S_grad)
trace_matrix(S_grad, TF_grad, interface).transpmult(Tstress.vector(), stress_s.vector())

es = [inner(dot(grad(uS0), n) - dot(stress_s, n),
            dot(grad(uS0), n) - dot(stress_s, n))*dI_s,
      inner(grad(uS0) - stress_s, grad(uS0) - stress_s)*dI_s]

for e in es:
    print sqrt(abs(assemble(e)))
# Now we're close enough again
