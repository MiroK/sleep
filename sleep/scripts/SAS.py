### SAS
# simplified model of the subarachnoid space.
# cylindrical symmetry
# mesh : a sphere for cranium SAS + cylinder for spinal SAS


#todo : outflow granulation dependent on the pressure
# check the pressure evolution
# non oscillating case without injection
# non oscillating case with injection : like the infusion test --> check the rate in Eric experiments


from sleep.mesh.dolfinconvert import convert
from sleep.mesh import load_mesh2d
from sleep.fbb_DD.fluid import solve_fluid_cyl as solve_fluid
from sleep.fbb_DD.advection import solve_adv_diff_cyl as solve_adv_diff
import dolfin as df
import math
import os

meshdir='../mesh/gmsh_export/'
# output folder name
outputfolder='../output/SAS_injection_HB/'


tfinal=5*60
toutput=1
dt=1/10/8

#Geometry
r_in=3e-1
r_spinal=0.5e-1
h=0.1e-1
xmax=8e-1

# tracer
D=2e-8
sigma_gauss=0.02
xi_gauss=(2.875e-1,1.68e-1)

if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

if not os.path.exists(outputfolder+'/fields'):
        os.makedirs(outputfolder+'/fields')

#pvd files
uf_out, pf_out= df.File(outputfolder+'fields'+'/uf.pvd'), df.File(outputfolder+'fields'+'/pf.pvd')
c_out= df.File(outputfolder+'fields'+'/c.pvd')

# load mesh
gmsh_file=meshdir+"/SAS_medium_injection.msh"
h5_file=convert(gmsh_file)

mesh = df.Mesh()


h5 = df.HDF5File(mesh.mpi_comm(), h5_file, 'r')
h5.read(mesh, 'mesh', False)

facet_f= df.MeshFunction("size_t", mesh, mesh.topology().dim()-1,0)

#The mesh is in mm, convertion into cm
xy = mesh.coordinates()             
scaling_factor = 0.1
xy[:, :] *= scaling_factor
mesh.bounding_box_tree().build(mesh) 


# Label facets
xy = mesh.coordinates().copy()              
x, y= xy.T


tol=h/10

class Boundary_skullleft(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary  and df.near(x[1],0,tol) and (x[0] > -r_in-h-tol)  and (x[0] < -r_in+tol)

class Boundary_skullright(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary  and df.near(x[1],0,tol) and (x[0] > r_in-tol)  and (x[0] < r_in+h+tol)

class Boundary_skullbottom(df.SubDomain):
    def inside(self, x, on_boundary):
        r = math.sqrt(x[0] * x[0] + x[1] * x[1])
        return on_boundary  and df.near(r,r_in,tol)

class Boundary_skulltop(df.SubDomain):
    def inside(self, x, on_boundary):
        r = math.sqrt(x[0] * x[0] + x[1] * x[1])
        return on_boundary  and df.near(r,r_in+h,tol)

class Boundary_spinalbottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary  and df.near(x[1],r_spinal,tol) and (x[0] > math.sqrt(r_in**2-r_spinal**2)-tol) 

class Boundary_spinaltop(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary  and df.near(x[1],r_spinal+h,tol) and (x[0] > math.sqrt(r_in**2-(r_spinal+h)**2)-tol) 

class Boundary_spinalright(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary  and df.near(x[0],xmax,tol) 

# Location for tracer injection

dx_tracer=0.05e-1

class Boundary_injection(df.SubDomain):
    def inside(self, x, on_boundary):
        r = math.sqrt(x[0] * x[0] + x[1] * x[1])
        return on_boundary  and df.near(r,r_in+h,tol) and (x[0] > 2.85e-1)  and (x[0] < 2.85e-1+dx_tracer) and (x[0] >0)




bskullleft= Boundary_skullleft()
bskullright= Boundary_skullright()
bskullbottom =  Boundary_skullbottom()
bskulltop =  Boundary_skulltop()

bspinalright= Boundary_spinalright()
bspinalbottom =  Boundary_spinalbottom()
bspinaltop =  Boundary_spinaltop()

binjection =  Boundary_injection()

bskullleft.mark(facet_f, 1) 
bskullright .mark(facet_f,  2) 
bskullbottom.mark(facet_f,  5) 
bskulltop.mark(facet_f,  3) 

bspinalright.mark(facet_f, 4) 
bspinalbottom.mark(facet_f,  3) 
bspinaltop.mark(facet_f,  3) 

binjection.mark(facet_f,  6) 

facet_lookup = {'source': 2 ,'sink': 1, 'noflow': 3, 'out':4, 'brain':5, 'injection':6}

#translation
xy = mesh.coordinates()   
xy[:, 1] += 0.5e-1
mesh.bounding_box_tree().build(mesh) 

df.File('./meshview/fbb_facets.pvd') << facet_f

## Define boundary conditions

#Brain outflow due to vessel contraction/dilation
blood_vol=12.5e-3  #cm3 total cerebral blood volume
brainvolume= 400e-3 #cm3
blood_vol=0.035*brainvolume
pc_arterioles=0.2 #pc of total blood volume in arterioles
V0_arterioles=pc_arterioles*blood_vol
amp=0.02 # ratio of volume change in the arterioles
w=2*math.pi*10 #frequency
#Varterioles=V0(1+asin(wt))
#Qsurf=dVarterioles/dt
#Qsurf=V0*a*cos(wt)
#Usurf=Qsurf/Area=Qsurf/(4*pi*r^2)
U0=amp*V0_arterioles/(4*math.pi*r_in**2)

u_brain = df.Expression(('u*cos(w*time)*cos(acos(x[0]/r))','u*cos(w*time)*sin(acos(x[0]/r))'),u=U0, w=w, r=r_in*(1+h/10), time=0.0, degree=2)
# I set r=r_in +h/10 because if not a get na, probably because of tol in x value and acos maube not define for x/r=1

# injection of tracer over the first 2.5 minutes at a rate of 1ul/min

z, r = df.SpatialCoordinate(mesh)
ds = df.Measure('ds', domain=mesh, subdomain_data=facet_f)
surf_injection=df.assemble(2*math.pi*r*ds(facet_lookup['injection']))
#surf_injection=2*math.pi*(r_in+h)*dx_tracer
q_injection=1e-3/60
u_injection=df.Expression(('time < tend ? u*cos(acos(x[0]/r)) : 0 ','time < tend ?u*sin(acos(x[0]/r)) : 0'),u=-q_injection/surf_injection, r=(r_in+h), tend=1*60, time=0.0, degree=2)

c_injection=df.Expression('time < tend ? 1 : 0 ', tend=1*60, time=0.0, degree=0)


print('u injection',q_injection/surf_injection)

# CSF production
q=6e-3 #mm3/s
q=q*1e-3 #cm3/s
# continuous production
#surface : cylinder radius 0.5e-1 thickness h
#surf=0.5e-1*h*2*math.pi
#u_prod=q/surf


# CSF absorption
q=6e-3 #mm3/s
q_prod=q*1e-3 #cm3/s
#surface : cylinder radius 0.5e-1 thickness h
surf_sink=0.5e-1*h*2*math.pi
surf_source=0.5e-1*h*2*math.pi
#surf_sink=df.assemble(2*math.pi*r*ds(facet_lookup['sink']))
#surf_source=df.assemble(2*math.pi*r*ds(facet_lookup['source']))
u_abs=df.Expression((0,'-q/surf'), q=q_prod, surf=surf_sink, degree=0)
u_prod=df.Expression((0,'q/surf'), q=q_prod, surf=surf_source, degree=0)


#output pressure in spinal canal
# Compliance law dP/dt=dP/dV.dV/dt=dP/dV.qout=E*P*qout
#Pn+1=pn+dt*(E*P*qoutn)
E=100
P0=5330
Pn=5330
Pout=df.Expression('Pn', Pn=Pn, degree=0) #


bcs_tracer = {'concentration': [(facet_lookup['source'], df.Constant(0)),
                                (facet_lookup['sink'], df.Constant(0)),
                                (facet_lookup['injection'],c_injection)
                                ],
                    'flux': [(facet_lookup['noflow'], df.Constant(0)),
                            (facet_lookup['brain'], df.Constant(0)),
                            (facet_lookup['out'], df.Constant(0))]}

bcs_fluid = {'velocity': [(facet_lookup['source'],u_prod),
                          (facet_lookup['sink'], u_abs),
                          (facet_lookup['noflow'], df.Constant((0,0))),
                          (facet_lookup['brain'], u_brain),
                          (facet_lookup['injection'], u_injection)],
            'traction': [],  
            'pressure': [(facet_lookup['out'], Pout)]}

fluid_parameters = {'mu': 7e-3, 'rho': 1, 'dt':dt}
tracer_parameters={'kappa': D, 'dt':dt, 'nsteps':1}



#FEM space

Vf_elm = df.VectorElement('Lagrange', df.triangle, 2)
Qf_elm = df.FiniteElement('Lagrange', df.triangle, 1)
Wf_elm = df.MixedElement([Vf_elm, Qf_elm])
Wf = df.FunctionSpace(mesh, Wf_elm)

Ct_elm = df.FiniteElement('Lagrange', df.triangle, 1)
Ct = df.FunctionSpace(mesh, Ct_elm)

# Initialisation : 
uf_n = df.project(df.Constant((0, 0)), Wf.sub(0).collapse())
pf_n =  df.project(df.Constant(5330), Wf.sub(1).collapse())

c_0 = df.Expression('exp(-a*pow(x[0]-xi_x, 2)-a*pow(x[1]-xi_y, 2)) ', degree=1, a=1/2/sigma_gauss**2, xi_x=xi_gauss[0], xi_y=xi_gauss[1])
c_n =  df.project(c_0,Ct)
#c_n =  df.project(df.Constant(0),Ct)

# Save initial state
uf_n.rename("uf", "tmp")
pf_n.rename("pf", "tmp")
c_n.rename("c", "tmp")
uf_out << (uf_n, 0)
pf_out << (pf_n, 0)
c_out << (c_n, 0)

# Time loop
time = 0.
timestep=0

intQout=0

Pc0=30*1333 # venous pressure

Rout=(3*1333)/(3*1e-3/60) # governed by the Delta P pressure for injection at rate 3 ul/min

Pss=-q_prod*Rout+P0

Rprod=(Pc0-P0)/q_prod # dyn/cm5 s

while time < tfinal:


    #update BC
    u_brain.time = time
    c_injection.time=time
    u_injection.time=time

    # outflow at granulation
    # mean pressure
    z, r = df.SpatialCoordinate(mesh)
    ds = df.Measure('ds', domain=mesh, subdomain_data=facet_f)
    Psas=df.assemble(2*math.pi*r*pf_n*ds(facet_lookup['sink']))/surf_sink
    Q_sink=(Psas-Pss)/Rout

    # inflow from production
    Pc=Pc0*(1+5*1333*math.sin(2*math.pi*10*time)) # with heart beat
    Psas=df.assemble(2*math.pi*r*pf_n*ds(facet_lookup['source']))/surf_source
    Q_source=(Pc-Psas)/Rprod   

    u_prod.q=Q_source
    u_abs.q=Q_sink

    # Compliance law dP/dt=dP/dV.dV/dt=dP/dV.qout=E*P*qout
    
    #qout=2 pi int_Rv^Rpvs u r dr 
    z, r = df.SpatialCoordinate(mesh)
    ds = df.Measure('ds', domain=mesh, subdomain_data=facet_f)
    n = df.FacetNormal(mesh)
    Qout=df.assemble(2*math.pi*r*df.dot(uf_n, n)*ds(facet_lookup['out']))
    #Pn+1=pn+dt*(E*P*qoutn)
    #Pout.Pn=Pn+dt*E*Pn*Qout
    # dP/dt=dP/dV.dV/dt=dP/dV.qout=E*P*qout
    # ln(p)-ln(p0)=E int_0^t Qout
    #p=p0exp(E int_0^t Qout) 
    intQout+=Qout*dt
    Pout.Pn=P0*math.exp(E*intQout)


    print('Qout',Qout)
    print('integral Qout',intQout)
    

    


    # Solve fluid problem
    uf_, pf_ = solve_fluid(Wf, u_0=uf_n,  f=df.Constant((0, 0)), bdries=facet_f, bcs=bcs_fluid,
                            parameters=fluid_parameters)

    tracer_parameters["T0"]=time

    # Solve tracer problem
    c_, T0= solve_adv_diff(Ct, velocity=uf_, f=df.Constant(0), phi_0=c_n,
                                  bdries=facet_f, bcs=bcs_tracer, parameters=tracer_parameters)

    # Update current solution
    uf_n.assign(uf_)
    pf_n.assign(pf_)
    c_n.assign(c_)

    Pn=Pout.Pn

    #Update time
    time += dt
    timestep+=1

    # Save output
    if(timestep % int(toutput/dt) == 0):

        uf_.rename("uf", "tmp")
        pf_.rename("pf", "tmp")
        c_.rename("c", "tmp")
        uf_out << (uf_, time)
        pf_out << (pf_, time)
        c_out << (c_, time)




