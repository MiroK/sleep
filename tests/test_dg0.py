from sleep.utils import trace_matrix
from sleep.utils import EmbeddedMesh
from dolfin import *
import numpy as np

class Foo(Expression):
    def eval_cell(self, values, x, ufc_cell):
        values[0] = ufc_cell.index

f = Foo(degree=0)

mesh = UnitSquareMesh(32, 32)
facet_f = MeshFunction('size_t', mesh, 1, 0)

DomainBoundary().mark(facet_f, 1)
Tmesh = EmbeddedMesh(facet_f, 1)

V = FunctionSpace(mesh, 'DG', 0)
u = interpolate(f, V)

TV = FunctionSpace(Tmesh, 'DG', 0)
Tu = Function(TV)
trace_matrix(V, TV, Tmesh).mult(u.vector(), Tu.vector())

assert assemble(inner(f - Tu, f - Tu)*ds) < 1E-13
