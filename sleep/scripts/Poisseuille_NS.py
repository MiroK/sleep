# This script aims to compute the flow between two concentric cylinder with an impose pressure gradient
# This is a poisseuille flow where the analytical solution is known
# We use the NS fluid solver
# Fluid parameters values corresponds to the PVS problem
# Pressure difference : 1 mmHg = 1330 dyn/cm2

from sleep.fbb_DD.domain_transfer import transfer_into
from sleep.fbb_DD.solid import solve_solid
from sleep.fbb_DD.fluid_NS import solve_fluid
from sleep.fbb_DD.ale import solve_ale
from sleep.utils import EmbeddedMesh
from sleep.mesh import load_mesh2d
from dolfin import *
from mshr import Polygon, generate_mesh

#time parameters
Toutput=1e-2
tfinal=5e-2
dt=1e-4

# Geometry params
Rv = 30e-4 # centimeters
Rpvs =40e-4 # centimeters
L = 100e-4 # centimeters

#(4, 8, 16, 32, 64)
N=8

# BC parameter
delta_P=1330 # dyn/cm2

# material parameters
mu=7e-3 #dyn·s/cm2
rho=1 # gram7cm3

# output folder name
outputfolder='Poisseuille_NS'

#FEM space
Vf_elm = VectorElement('Lagrange', triangle, 2)
Qf_elm = FiniteElement('Lagrange', triangle, 1)

# Exact solutions
dPdx=delta_P/L
h=(Rpvs-Rv)/2
umax=-dPdx*h**2/(2*mu)
#An Expression will be interpolated into a finite element space before the quadrature is carried out. 
#The degree argument to the Expression constructor is the degree of the FE space.  degree= ...
#Alternatively, you can specify the element
p_exact = Expression('dPdx*x[0]', dPdx=dPdx, degree=3)

# cylindrical : delta_P/L/4/mu*(std::log(x[1]/rOut)/std::log(rOut/rIn)*(rOut*rOut - rIn*rIn) + (rOut*rOut - x[1]*x[1]))
u_exact = Expression(('umax*(1-pow((x[1]-(ymin+ymax)/2),2)/pow(h,2))', '0'), umax=umax, h=h,ymin=Rv,ymax=Rpvs, degree=4)


# Create output folders
uf_out, pf_out= File('./output/'+outputfolder+'/uf.pvd'), File('./output/'+outputfolder+'/pf.pvd')
facets_out=File('./output/'+outputfolder+'/facets.pvd')

#Geometry computations
#domain_vertices = [Point(0, Rv),Point(L, Rv),Point(L, Rpvs),Point(0, Rpvs)]

# Meshing 
#domain = Polygon(domain_vertices)
#mesh_f = generate_mesh(domain,resolution)

mesh_f = RectangleMesh(Point(0, Rv), Point(L, Rpvs), 10*N, N)

fluid_bdries = MeshFunction("size_t", mesh_f, mesh_f.topology().dim()-1,0)

# Label facets
xy = mesh_f.coordinates().copy()              
x, y = xy.T

xmin=x.min()
xmax=x.max()
ymin=y.min()
ymax=y.max()


tol=(Rpvs-Rv)/N/2 #cm

class Boundary_left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],xmin,tol) #left


class Boundary_right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],xmax,tol)# right


class Boundary_bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], ymin,tol) #bottom
    
class Boundary_top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], ymax,tol) #top



btop= Boundary_top()
bbottom = Boundary_bottom()
bleft = Boundary_left()
bright = Boundary_right()


bleft.mark(fluid_bdries, 1) 
bbottom.mark(fluid_bdries,  2) 
bright.mark(fluid_bdries,  3) 
btop.mark(fluid_bdries,  4)  
 

facet_lookup = {'x_min': 1 ,'y_min': 2, 'x_max': 3, 'y_max': 4}

facets_out << fluid_bdries

# Parameters setup ------------------------------------------------ FIXME

fluid_parameters = {'mu': mu, 'rho': rho, 'dt':dt}


# Setup fem spaces ---------------------------------------------------
Wf_elm = MixedElement([Vf_elm, Qf_elm])
Wf = FunctionSpace(mesh_f, Wf_elm)


# Setup of boundary conditions ----------------------------------- FIXME

import sympy
ts = sympy.symbols("time")

t1 = sympy.symbols("t1")
t2 = sympy.symbols("t2")
sin = sympy.sin

amp=1e-4 #cm
f=1 #Hz

functionU = amp*sin(2*pi*f*t1) # displacement
U_vessel = sympy.printing.ccode(functionU)

#functionV = amp*(sin(2*pi*f*t2)-sin(2*pi*f*t1))/(t2-t1) # velocity
functionV = sympy.diff(functionU,t1) # velocity
V_vessel = sympy.printing.ccode(functionV)

functionUALE=amp*(sin(2*pi*f*t2)-sin(2*pi*f*t1))
UALE_vessel = sympy.printing.ccode(functionUALE)


# Now we wire up
bcs_fluid = {'velocity': [(facet_lookup['y_min'], Constant((0,0))),
                          (facet_lookup['y_max'], Constant((0,0)))],
             'traction': [],  
             'pressure': [(facet_lookup['x_min'], Constant(0)),
                          (facet_lookup['x_max'], Constant(delta_P))]}







# Define functions for solutions at previous and current time steps
#  Initialise with analytical solution
uf_n = project(Constant((0, 0)), Wf.sub(0).collapse())
pf_n =  project(Constant(0), Wf.sub(1).collapse())
#uf_n = project(u_exact, Wf.sub(0).collapse())
#pf_n =  project(p_exact, Wf.sub(1).collapse())

#add random perturbation
#eps=0.01
#from numpy import random
#import numpy as np
#uf_n.vector().set_local(np.array(uf_n.vector())*(1+eps*(0.5-random.random(uf_n.vector().size()))))
#pf_n.vector().set_local(np.array(pf_n.vector())*(1+eps*(0.5-random.random(pf_n.vector().size()))))

# Time loop
time = 0.
timestep=0




# Save initial state
uf_n.rename("uf", "tmp")
pf_n.rename("pf", "tmp")

uf_out << (uf_n, time)
pf_out << (pf_n, time)

erroru=[]
errorp=[]
while time < tfinal:
    time += fluid_parameters['dt']
    timestep+=1

    # Solve fluid problem
    uf_, pf_ = solve_fluid(Wf, f=Constant((0, 0)), u_n=uf_n, p_n=pf_n, bdries=fluid_bdries, bcs=bcs_fluid,
                           parameters=fluid_parameters)

    # Update current solution
    uf_n.assign(uf_)
    pf_n.assign(pf_)

    # Errors
    eu = errornorm(u_exact, uf_, 'H1', degree_rise=2)
    ep = errornorm(p_exact, pf_, 'L2', degree_rise=2)

    print('|u-uh|_1', eu, '|p-ph|_0', ep)
    
    # Save output
    if(timestep % int(Toutput/fluid_parameters['dt']) == 0):

        erroru.append(eu)
        errorp.append(ep)

        uf_.rename("uf", "tmp")
        pf_.rename("pf", "tmp")

        uf_out << (uf_, time)
        pf_out << (pf_, time)

with open('./output/'+outputfolder+'/erroru.txt', 'a') as csv:
    row=[N]+erroru
    csv.write(('%i'+', %e'*len(erroru)+'\n')%tuple(row))

with open('./output/'+outputfolder+'/errorp.txt', 'a') as csv:
    row=[N]+errorp
    csv.write(('%i'+', %e'*len(errorp)+'\n')%tuple(row))