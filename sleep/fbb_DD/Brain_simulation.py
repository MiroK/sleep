#! /usr/bin/env python3
import argparse
import logging
from datetime import datetime
import os

import numpy as np
from math import pi, ceil

from sleep.fbb_DD.domain_transfer import transfer_into
from sleep.fbb_DD.advection import solve_adv_diff as solve_adv_diff
#from sleep.fbb_DD.fluid import solve_fluid as solve_fluid
from sleep.fbb_DD.fluid import solve_fluid
#import sleep.fbb_DD.cylindrical as cyl
#todo : cylindrical for solid
from sleep.fbb_DD.solid2 import solve_solid
from sleep.fbb_DD.ale import solve_ale as solve_ale
from sleep.utils import EmbeddedMesh
from sleep.mesh import load_mesh2d
from dolfin import *

import gmsh

#Todo : implement the other version for the mesh, with lateral progression also

# What are the conditions for time step ?
# All implicit so the only worry is numerical diffusion.
#todo : add dt and n steps management


# for the resistance BC : It does not work fo high resistance (ex R=1e11), we cannot tend to zero flow case. 
# Should we iterate ? 
# Need to analyse for time step convergence. 


# Define line function for 1D slice evaluation
def line_sample(line, f, fill=np.nan):
    values = fill*np.ones(len(line))
    for i, xi in enumerate(line):
        try:
            values[i] = f(xi)
        except RuntimeError:
            continue
    return values

def slice_integrate_cyl(x,f,ymin,ymax,N=10) :
    vertical_line=line([x,ymin],[x,ymax], N)
    values=line_sample(vertical_line, f)
    r=np.linspace(ymin,ymax,N)
    integral=2*np.trapz(values*r,r)/(ymax**2-ymin**2) #cylindrical coordinate
    return integral

def profile_cyl(f,xmin,xmax,ymin,ymax,Nx=100,Ny=10):
    spanx=np.linspace(xmin,xmax,Nx)
    values=[slice_integrate_cyl(x,f,ymin,ymax,N=Ny) for x in spanx]
    return np.array(values)



def line(A, B, nsteps):
    A=np.array(A)
    B=np.array(B)
    return A + (B-A)*np.linspace(0, 1, nsteps).reshape((-1, 1))


def title1(string):
    line1='\n'+'*'*100+'\n'
    line2='**   '+string+'\n'
    line3='*'*100
    return line1 + line2 + line3



def PVSbrain_simulation(args):
    """ Solve the flow and tracer transport in the brain only(Biot domain) :
    Outputs :
    - a logfile with information about the simulation
    - .pvd files at specified args.toutput time period with the u, p and c fields in stokes domain and u, p , q, phi and c fields in Biot domain
    - .csv files of u, p, c 1D array of the u, p, c fields on the middle line of the PVS 
    - .csv files of the total mass in the domain, mass flux from the brain to sas, brain to PVS, PVS to SAS

    """



    # output folder name
    outputfolder=args.output_folder+'/'+args.job_name+'/'

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)


    if not os.path.exists(outputfolder+'/profiles'):
        os.makedirs(outputfolder+'/profiles')

    if not os.path.exists(outputfolder+'/fields'):
        os.makedirs(outputfolder+'/fields')

    # Create output files

    #txt files
    csv_p=open(outputfolder+'profiles'+'/pressure.txt', 'w')
    csv_u=open(outputfolder+'profiles'+'/velocity.txt', 'w')
    csv_c=open(outputfolder+'profiles'+'/concentration.txt', 'w')
    csv_rv=open(outputfolder+'profiles'+'/radius.txt', 'w')

    csv_mass=open(outputfolder+'profiles'+'/mass.txt', 'w')
    #time,  total mass in the domain, mass flux from the brain to sas, brain to PVS, PVS to SAS

    #pvd files
    uf_out, pf_out= File(outputfolder+'fields'+'/uf.pvd'), File(outputfolder+'fields'+'/pf.pvd') 
    etas_out, qs_out, ps_out , = File(outputfolder+'fields'+'/us.pvd'), File(outputfolder+'fields'+'/qs.pvd'), File(outputfolder+'fields'+'/ps.pvd') 
    c_out, phi_out=File(outputfolder+'fields'+'/c.pvd'),File(outputfolder+'fields'+'/phis.pvd')
   
    facets_out_fluid=File(outputfolder+'fields'+'/facets_fluid.pvd')
    facets_out_solid=File(outputfolder+'fields'+'/facets_solid.pvd')

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # log to a file
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(outputfolder+'/', 'PVSBrain_info.log')
    file_handler = logging.FileHandler(filename,mode='w')
    file_handler.setLevel(logging.INFO)
    #formatter = logging.Formatter("%(asctime)s %(filename)s, %(lineno)d, %(funcName)s: %(message)s")
    #file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # log to the console
    console_handler = logging.StreamHandler()
    level = logging.INFO
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    

    # initialise logging

    logging.info(title1("Simulation of the PVS + Brain flow, deformation and tracer transport using stockes solver, biot solver and diffusion-advection solver"))

    logging.info("Date and time:"+datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

    logging.info('Job name : '+args.job_name)

    logging.debug('logging initialized')


    # Set parameters

    logging.info(title1("Parameters"))

    # Geometry params
    logging.info('\n * Geometry')
    Rv = args.radius_vessel # centimeters
    Rpvs =args.radius_pvs # centimeters 
    Rbrain =args.radius_brain # centimeters
    L = args.length # centimeters



    logging.info('PVS radius  : %e cm'%Rpvs)
    logging.info('Brain radius : %e cm'%Rbrain)
    logging.info('length : %e cm'%L)

    #Mesh
    logging.info('\n * Mesh')
    #number of cells in the radial direction fluid domain
    Nr=args.N_radial_fluid
    #number of cells in the radial direction biot domain
    Nr_biot=args.N_radial_biot
    s_biot=args.biot_progression

    DR=(Rpvs-Rv)/Nr
    #number of cells in the axial direction
    if args.N_axial :
        Nl=args.N_axial
    else :
        Nl=round(L/DR)

    DY=L/Nl
    
    logging.info('N axial: %i'%Nl)
    logging.info('N radial Biot: %e'%Nr_biot)
    logging.info('progression parameter in biot: %e'%s_biot)

    
    #Oscillation parameters

    logging.info('\n * Cross section area parameters')
    ai=args.ai
    fi=args.fi
    phii=args.phii
    logging.info('ai (dimensionless): '+'%e '*len(ai)%tuple(ai))
    logging.info('fi (Hz) : '+'%e '*len(fi)%tuple(fi))
    logging.info('phii (rad) : '+'%e '*len(phii)%tuple(phii))


    #time parameters
    logging.info('\n * Time')
    toutput=args.toutput
    toutput_cycle=args.toutputcycle
    tfinal=args.tend



    ## Here we take a time step that corresponds to the period
    # Todo adapt when fi is a list : find the global period
    if fi[0]==0 :
        period=tfinal
        dt=args.time_step
    else :
        period=1/fi[0]
        dt=args.time_step
        # at least 4 timestep per period 
        Ntimesteps=max(4,int(period/dt))
        dt=period/Ntimesteps

    # at least 4 output for the cycle
    n_output=max(4,int(period/toutput_cycle))
    toutput_cycle=period/n_output

    # at least the time step
    toutput=max(toutput,dt)


    ## Number of period to solve
    N_cycles=ceil(tfinal/period)

    logging.info('final time: %e s'%tfinal)
    logging.info('output period during the stokes biot resolution : %e s'%toutput_cycle)
    logging.info('output period during the advection diffusion resolution : %e s'%toutput)
    logging.info('time step requested: %e s'%args.time_step)
    logging.info('oscillation period: %e s'%period)
    logging.info('Nb of cycles : %e s'%dt)

    # approximate CFL for fluid solver : need to compute max velocity depending on the wall displacement... 
    # maybe just add a warning in computation with actual velocity
    #Uapprox=500e-4 #upper limit for extected max velocity
    #CFL_dt=0.25*DY/Uapprox
    #if  CFL_dt < dt :
    #    logging.warning('The specified time step of %.2e s does not fullfil the fluid CFL condition. New fluid time step : %.2e s'%(dt, CFL_dt))
    #dt_fluid=min(dt,CFL_dt)
    dt_fluid=dt

    # approximate CFL for tracer solver
    dt_advdiff=dt


    # material parameters
    logging.info('\n * Fluid properties')
    mu=args.viscosity
    rho=args.density

    logging.info('density: %e g/cm3'%rho)
    logging.info('dynamic viscosity : %e dyn s/cm2'%mu)

    logging.info('\n* Tracer properties')
    D=args.diffusion_coef
    sigma_gauss=args.sigma
    logging.info('Free diffusion coef: %e cm2/s'%D)
    logging.info('STD of initial gaussian profile: %e '%sigma_gauss)
    xi_gauss=args.initial_pos
    logging.info('Initial position: %e cm2'%xi_gauss)



    logging.info('\n * Porous medium properties')

    permeability=args.biot_permeability
    conductivity=args.biot_permeability/mu
    E=args.biot_youngmodulus
    poisson=args.biot_poisson
    alpha=args.biot_biotcoefficient
    porosity_0=args.biot_porosity
    tortuosity=args.biot_tortuosity

    G=E/(2*(1+poisson))
    lmbda=E*poisson/(1+poisson)/(1-2*poisson)

    s0=1/lmbda

    dt_solid=dt



    logging.info('initial porosity: %e '%porosity_0)
    logging.info('tortuosity: %e '%tortuosity)
    logging.info('hydraulic conductivity : %e cm2/(g.cm-1.s-1)'%conductivity)
    logging.info('permeability : %e cm2'%permeability)
    logging.info('poisson coefficient : %e '%poisson)
    logging.info('biot coefficient : %e '%alpha)
    logging.info('storage coefficient : %e '%s0)
    logging.info('Young modulus : %e dyn s/cm2'%E)
    logging.info('Shear modulus : %e dyn s/cm2'%G)
    logging.info('Lambda Lame : %e dyn s/cm2'%lmbda)



    logging.info('\n * ALE')
    k_ale=args.ale_parameter
    logging.info('ALE parameter: %e '%k_ale)

    logging.info('\n * Lateral BC')
    resistance=args.resistance
    logging.info('inner resistance: %e '%resistance)
    if resistance == 0 :
        lateral_bc='free'
        logging.info('right BC will be set to the free assumption')
    elif resistance < 0 :
        lateral_bc='noflow'
        logging.info('right BC will be set to the no flow assumption')
    else :
        lateral_bc='resistance'
        logging.info('right BC will be set to the resistance assumption')

    fluid_parameters = {'mu': mu, 'rho': rho, 'dt':dt_fluid}
    ale_parameters = {'kappa': k_ale}
    solid_parameters = {'kappa': conductivity, 'mu': G, 'lmbda': lmbda, 
                    'alpha': alpha, 's0': s0, 'dt':dt_solid}  



    ## The tracer is solver in the full domain, so it has two subdomains
    # 1 for solid
    # 0 for fluid

    tracer_parameters={'kappa_0': D, 'kappa_1': D*tortuosity,'dt':dt_advdiff, 'nsteps':1}



    # Mesh
    logging.info(title1('Meshing'))

    meshing=args.mesh_method

    if meshing=='regular' :
        # Create a mesh using Rectangle mesh : all the cells are regulars but this means a lot of cells

        logging.info('cell size : %e cm'%(np.sqrt(DR**2+DY**2)))

        # Create a rectangle mesh with Nr + Nsolid cells in the radias direction and Nl cells in the axial direction
        # the geometrical progression for the solid mesh is s
        

        #Creation of the uniform mesh
        Rext=1+Nr_biot/Nr
        mesh= RectangleMesh(Point(0,0), Point(L, Rext), Nl, Nr+Nr_biot)
        x = mesh.coordinates()[:,0]
        y = mesh.coordinates()[:,1]

        #Deformation of the mesh

        def deform_mesh(x, y):
            transform_fluid=Rv + (Rpvs-Rv)*y
            transform_solid=Rpvs + (Rbrain-Rpvs)*((y-1)/(Rext-1))**s_biot
            yp=np.where(y <= 1, transform_fluid, transform_solid)
            return [x, yp]

        x_bar, y_bar = deform_mesh(x, y)
        xy_bar_coor = np.array([x_bar, y_bar]).transpose()
        mesh.coordinates()[:] = xy_bar_coor
        mesh.bounding_box_tree().build(mesh)

    else :

        
        from sleep.mesh import mesh_model2d, load_mesh2d, set_mesh_size
        import sys


        gmsh.initialize(['','-format', 'msh2'])


        model = gmsh.model

        import math
        Apvs0=math.pi*Rpvs**2
        Av0=math.pi*Rv**2
        A0=Apvs0-Av0


        # progressive mesh 
        factory = model.occ
        a = factory.addPoint(0, Rv,0)
        b = factory.addPoint(L,Rv,0)
        c = factory.addPoint(L,Rpvs,0)
        d = factory.addPoint(0,Rpvs,0)
        e = factory.addPoint(L,Rbrain,0)
        f = factory.addPoint(0,Rbrain,0)


        fluid_lines = [factory.addLine(*p) for p in ((d,a),(a,b), (b, c),(c,d) )]
        interface=[fluid_lines.pop()]
        named_lines = dict(zip(('pvs_left','bottom', 'pvs_right'), fluid_lines))


        solid_lines = [factory.addLine(*p) for p in ((c, e), (e,f), (f,d))]
        named_lines.update(dict(zip(( 'brain_right', 'brain_top', 'brain_left'), solid_lines)))

        named_lines.update(dict(zip(( 'interface'), interface)))

        fluid_loop = factory.addCurveLoop(fluid_lines+interface)
        fluid = factory.addPlaneSurface([fluid_loop])

        solid_loop = factory.addCurveLoop(interface+solid_lines)
        solid = factory.addPlaneSurface([solid_loop])


        factory.synchronize()

        tags = {'cell': {'F': 1, 'S': 2},
            'facet': {}}

        model.addPhysicalGroup(2, [fluid], 1)
        model.addPhysicalGroup(2, [solid], 2)

        for name in named_lines:
            tag = named_lines[name]
            model.addPhysicalGroup(1, [tag], tag)

        # boxes for mesh refinement
        cell_size=DR*(Rpvs-Rv)/(Rpvs-Rv)
        boxes = []
        # add box on the PVS for mesh
        field = model.mesh.field


        fid = 1
        field.add('Box', fid)
        field.setNumber(fid, 'XMin', 0)
        field.setNumber(fid, 'XMax', L)
        field.setNumber(fid, 'YMin', Rv)
        field.setNumber(fid, 'YMax', Rpvs)
        field.setNumber(fid, 'VIn', cell_size)
        field.setNumber(fid, 'VOut', DR*50 )
        field.setNumber(fid, 'Thickness', (Rpvs-Rv)/4)

        boxes.append(fid)




        
        # Combine
        field.add('Min', fid+1)
        field.setNumbers(fid+1, 'FieldsList', boxes)    
        field.setAsBackgroundMesh(fid+1)

        model.occ.synchronize()

        h5_filename = outputfolder+'/mesh.h5'

        mesh_model2d(model, tags, h5_filename)

        mesh, markers, lookup = load_mesh2d(h5_filename)
        
        gmsh.finalize()

    ## Define subdomains

    x = mesh.coordinates()[:,0]
    y = mesh.coordinates()[:,1]


    tol=1e-7


    class Omega_0(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] < Rpvs + tol

    class Omega_1(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] > Rpvs - tol

    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(),0)

    subdomain_0 = Omega_0()
    subdomain_1 = Omega_1()
    subdomain_0.mark(subdomains, 0)
    subdomain_1.mark(subdomains, 1)


    mesh_f = EmbeddedMesh(subdomains, 0)
    mesh_s = EmbeddedMesh(subdomains, 1)



    ## Define boundaries
    solid_bdries = MeshFunction("size_t", mesh_s, mesh_s.topology().dim()-1,0)
    fluid_bdries = MeshFunction("size_t", mesh_f, mesh_f.topology().dim()-1,0)
    full_bdries= MeshFunction("size_t", mesh, mesh.topology().dim()-1,0)

    # Label facets
  

    class Boundary_left_fluid(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],0,tol) and (x[1] <= Rpvs + tol) #left fluid


    class Boundary_right_fluid(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],L,tol) and (x[1] <= Rpvs + tol) # right fluid


    class Boundary_left_solid(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],0,tol) and (x[1] >= Rpvs - tol) #left solid


    class Boundary_right_solid(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],L,tol) and (x[1] >= Rpvs - tol) # right solid


    class Boundary_bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Rv,tol) #bottom
        
    class Boundary_top(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Rbrain,tol) #top

    class Boundary_interface(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Rpvs,tol) #interface      


    #todo : keep separate F and S for right and left in the full bdries

    btop= Boundary_top()
    bbottom = Boundary_bottom()
    bleft_fluid= Boundary_left_fluid()
    bright_fluid = Boundary_right_fluid()
    bleft_solid = Boundary_left_solid()
    bright_solid = Boundary_right_solid()
    binterface = Boundary_interface()

    bbottom.mark(fluid_bdries,  2) 
    binterface.mark(fluid_bdries,  4)  
    bleft_fluid.mark(fluid_bdries, 1) 
    bright_fluid.mark(fluid_bdries,  3) 

    binterface.mark(solid_bdries,  4) 
    bright_solid.mark(solid_bdries,  5) 
    btop.mark(solid_bdries,  6) 
    bleft_solid.mark(solid_bdries,  7) 

    bleft_fluid.mark(full_bdries, 1) 
    bleft_solid.mark(full_bdries,  7)  
    bbottom.mark(full_bdries,  2) 
    bright_fluid.mark(full_bdries,  3) 
    bright_solid.mark(full_bdries,  5) 
    btop.mark(full_bdries,  6) 

    facet_lookup = {'F_left': 1 ,'F_bottom': 2, 'F_right': 3, 'Interface': 4,'S_left': 7 ,'S_top': 6, 'S_right': 5}

    facets_out_fluid << fluid_bdries

    facets_out_solid << solid_bdries

    ##Mask functions for the subdomains
    #class function_subdomains(UserExpression):
    #    def __init__(self, materials, k_0, k_1, **kwargs):
    #        super().__init__(**kwargs) # This part is new!
    #        self.materials = materials
    #        self.k_0 = k_0
    #        self.k_1 = k_1

    #    def eval_cell(self, values, x, cell):
    #        if self.materials[cell.index] == 0:
    #            values[0] = self.k_0
    #        else:
    #            values[0] = self.k_1
    #    def value_shape(self):
    #        #return (2,)
    #        return ()

    #is_solid = function_subdomains(subdomains, 0, 1, degree=0)
    #is_fluid = function_subdomains(subdomains, 1, 0, degree=0)


    # define the domain specific parameters for the tracer
    # NOTE: Here we do P0 projection
    dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
    CoefSpace = FunctionSpace(mesh, 'DG', 0)
    q = TestFunction(CoefSpace)

    # Remove
    coef='kappa'

    #fluid_coef = tracer_parameters.pop('%s_0' % coef)
    solid_coef = tracer_parameters.pop('%s_1' % coef)

    #form = ((1/CellVolume(mesh))*fluid_coef*q*dx(0) +
    #            (1/CellVolume(mesh))*solid_coef*q*dx(1))
        
    #tracer_parameters[coef] = Function(CoefSpace, assemble(form))

    tracer_parameters[coef]=Constant(solid_coef)

    #FEM space

    logging.info(title1("Set FEM spaces"))


    logging.info('\n * Porous medium')
    Es_elm = VectorElement('Lagrange', triangle, 2)
    Vs_elm = FiniteElement('Raviart-Thomas', triangle, 1)
    Qs_elm = FiniteElement('Discontinuous Lagrange', triangle, 0)
    Ws_elm = MixedElement([Es_elm, Vs_elm, Qs_elm])
    Ws = FunctionSpace(mesh_s, Ws_elm)

    logging.info('\n * Tracer')
    #### Todo : I would like to be able to have discontinuous concentration when we will have the membrane
    #### Beter to solve in two domains or one domain with discontinuous lagrange element ?
    Ct_elm = FiniteElement('Lagrange', triangle, 1)
    Ct = FunctionSpace(mesh_s, Ct_elm)
    logging.info('Concentration : "Lagrange", triangle, 1')


    logging.info('\n * ALE')
    # Displacement in the solid domain
    # Fenics won't let us move mesh with quadratic displacement so
    Va_s = VectorFunctionSpace(mesh_s, 'CG', 1)



    # Porosity variables 
    FS_porositys= FunctionSpace(mesh_s, 'DG', 0)

    #porosity_elm = FiniteElement('Lagrange', triangle, 1)
    #FS_porosity = FunctionSpace(mesh, porosity_elm)
    #FS_porositys =FunctionSpace(mesh_s, porosity_elm)

  


    # Setup of boundary conditions
    logging.info(title1("Boundary conditions"))

    import sympy
    tn = sympy.symbols("tn")
    tnp1 = sympy.symbols("tnp1")
    sin = sympy.sin
    sqrt = sympy.sqrt




    # PVS radius imposed as vessel radius

    functionR = Rpvs*(1+sum([a*sin(2*pi*f*tn+phi) for a,f,phi in zip(ai,fi,phii)])) # displacement
    R_vessel = sympy.printing.ccode(functionR)

    functionV = sympy.diff(functionR,tn) # velocity
    V_vessel = sympy.printing.ccode(functionV)

    #Delta U for ALE at the vessel boundary
    functionUALE=Rpvs*(1+sum([a*sin(2*pi*f*tnp1+phi) for a,f,phi in zip(ai,fi,phii)]))- Rpvs*(1+sum([a*sin(2*pi*f*tn+phi) for a,f,phi in zip(ai,fi,phii)]))
    UALE_vessel = sympy.printing.ccode(functionUALE)   

    vf_bottom = Expression(('0',V_vessel ), tn = 0, degree=2)   # no slip no gap condition at vessel wall 
    uale_bottom = Expression(('0',UALE_vessel ), tn = 0, tnp1=1, degree=2) # displacement for ALE at vessel wall 


    driving_expressions = (uale_bottom)


    logging.info('\n * Solid')
    logging.info('Left : zero pressure, stress free')
    logging.info('Right : no flow, no normal displacement')
    logging.info('Top : no flow, no normal displacement')
    logging.info('Bottom :Interface condition')


    ## to do : allow zero concentration if fluid BC is free on the right
    logging.info('\n * Tracer concentration')
    logging.info('Left : zero concentration')
    logging.info('Right : no flux')
    logging.info('Bottom: no flux')
    logging.info('Top: no flux')
    


    # Now we wire up


    #todo : add case fluid free
    bcs_tracer = {'concentration': [ (facet_lookup['S_left'], Constant(1))],
                    'flux': [(facet_lookup['S_right'], Constant(0)),
                            (facet_lookup['Interface'], Constant(0)),
                            (facet_lookup['S_top'], Constant(0))]}



    bcs_solid = {   'elasticity': {
                        'displacement': [(facet_lookup['Interface'], uale_bottom)],
                        'traction': [(facet_lookup['S_left'], Constant((0,0)))],
                        'displacement_x' : [(facet_lookup['S_right'],Constant((0)))],
                        'displacement_y' : [(facet_lookup['S_top'], Constant(0))]                         
                    },
                    'darcy': {
                        'pressure': [(facet_lookup['S_left'], Constant(0))],
                        'flux': [(facet_lookup['Interface'],  Constant((0,0))),
                                (facet_lookup['S_top'], Constant((0,0))),
                                (facet_lookup['S_right'], Constant((0,0)))]     
                    }
                }                   






    # Initialisation : 


    logging.info("\n * Porous medium")
    logging.info("Fluid velocity : zero field")
    logging.info("Fluid pressure : zero field")
    logging.info("Displacement : zero field")

    etas_n=project(Constant((0, 0)), Ws.sub(0).collapse())
    us_n=project(Constant((0, 0)), Ws.sub(1).collapse())
    ps_n=project(Constant(0), Ws.sub(2).collapse())

    etas_0 = project(Constant((0, 0)), Ws.sub(0).collapse()) # always at the begining of a time step

    

    logging.info("\n * Tracer")
    logging.info("Concentration : Gaussian profile")
    logging.info("                Centered at mid length")
    logging.info("                STD parameter = %e"%sigma_gauss)

    mesh.bounding_box_tree().build(mesh) 



    #Gaussian on the left side
    cf_0 = Expression('x[0]<= 50e-4 ? exp(-a*pow(x[0]-b, 2)): 0 ', degree=2, a=1/2/(2e-4)**2, b=0)

    #Gaussian on the right side
    #cf_0 = Expression('exp(-a*pow(x[0]-b, 2)) ', degree=2, a=1/2/(150e-4)**2, b=L, Rv=Rv, Rpvs=Rpvs)


    # 1 every where
    #cf_0=Constant(1)

    c_n =  project(cf_0,Ct)#


    porositys_n=project(Constant(porosity_0),FS_porositys)





    #Initial deformation of the fluid domain
    # We start at a time shift
    tshift= 0

    # todo if time shift : must modify the mesh 
 

    etas_n.rename("us", "tmp")
    us_n.rename("qs", "tmp")
    ps_n.rename("ps", "tmp")

    etas_out << (etas_n, 0)
    qs_out << (us_n, 0)
    ps_out << (ps_n, 0)

    c_n.rename("c", "tmp")
    c_out << (c_n, 0)

    porositys_n.rename("phi", "tmp")
    phi_out << (porositys_n, 0)



    ############# RUN ###########

    logging.info(title1("Run"))


    logging.info(title1("Biot over 2 cycles"))

    # Time loop
    time = tshift
    timestep=0


    ##############
    # First we solve the Biot stokes problem on 2 periods 
    # We dont take one period to avoid the effect of initial conditions.
    # We store the adv velocity and the deformation of the mesh in a file
    ##################

    advvel_file = HDF5File(mesh.mpi_comm(), outputfolder+"advection_velocity.h5", "w")
    meshdeformation_file= HDF5File(mesh.mpi_comm(), outputfolder+"mesh_deformation.h5", "w")
    porosity_file = HDF5File(mesh.mpi_comm(), outputfolder+"porosity.h5", "w")

    cycletimes=[]


    Nsteps=int(period/dt)


    for it in range(2*Nsteps+1):
        time=it*dt+tshift
        timestep+=1
        print('time', it*dt)

        # Update boundary conditions
        for expr in driving_expressions:
            hasattr(expr, 'tn') and setattr(expr, 'tn', time)
            hasattr(expr, 'tnp1') and setattr(expr, 'tnp1', time+dt)

        #move mesh 

        ALE.move(mesh_s, interpolate(etas_n, Va_s))
        mesh_s.bounding_box_tree().build(mesh_s)

        # Solve porous problem
        etas_, us_, ps_ = solve_solid(Ws, f1=Constant((0, 0)), f2=Constant(0), eta_0=etas_0, p_0=ps_n,
                                                bdries=solid_bdries, bcs=bcs_solid, parameters=solid_parameters)



        etas_n.assign(etas_)
        us_n.assign(us_)
        ps_n.assign(ps_)


        #### project on the whole mesh the advection velocity and porosity
        #porositys_=project(porositys_n+solid_parameters['alpha']*div(etas_)+solid_parameters['s0']*(ps_-ps_n),FS_porositys)
        porositys_=project((porositys_n+div(etas_))/(1+div(etas_)),FS_porositys)
        porositys_.set_allow_extrapolation(True)


        #CHECK : I use us_n to be sure to have continuity at the interface. I dont know if this 'delayed' adv velocity in the solid domain is an issue.
        porositys_n.assign(porositys_)

        ### Save the data needed for the adv_diff solver into h5 files
        ## On the second period
        if it>Nsteps:
            cycletimes.append((it-Nsteps)*dt)
            advvel_file.write(us_n.vector(), "/values_{}".format(len(cycletimes)))
            porosity_file.write(porositys_n.vector(), "/values_{}".format(len(cycletimes)))
            meshdeformation_file.write(etas_n.vector(), "/values_{}".format(len(cycletimes)))
            #cPickle.dump(cycletimes, open("times.cpickle", "w"))        # Save output

        ## Save the outputs in vtk format    
        if(timestep % int(toutput_cycle/dt) == 0):

            logging.info("\n*** save output time %e s"%(it*dt))
            logging.info("number of time steps %i"%(it*dt))

            # may report Courant number or other important values that indicate how is doing the run


            etas_n.rename("us", "tmp")
            us_n.rename("qs", "tmp")
            ps_n.rename("ps", "tmp")

            etas_out << (etas_n, it*dt)
            qs_out << (us_n, it*dt)
            ps_out << (ps_n, it*dt)


            porositys_n.rename("phi", "tmp")
            phi_out << (porositys_n, it*dt)



    logging.info(title1("Advection - diffusion up to the end "))

    # Restart time loop
    timestep=0

    n_cycle=-1
    timestep=0



    # Here I dont know if there will be several dt for advdiff and fluid solver
    while n_cycle < N_cycles-1:
        n_cycle+=1

        for it,time in enumerate(cycletimes) : 
            print(time)
            #get actual time
            current_time=n_cycle*period+time
            #get data from files
            advvel_file.read(us_n.vector(), "/values_{}".format(it+1), True)
            porosity_file.read(porositys_.vector(), "/values_{}".format(it+1), True)
            meshdeformation_file.read(etas_n.vector(), "/values_{}".format(it+1), True)

            timestep+=1
            print('time', current_time)


            #This is quite long
            ALE.move(mesh_s, etas_n)
            #is needed for the masks
            mesh.bounding_box_tree().build(mesh_s) 

            # Solve tracer problem
            # Todo : Maybe simplify and remove this as we are not using substeping ?
            tracer_parameters["T0"]=current_time


            c_, T0= solve_adv_diff(Ct, velocity=us_n, phi=porositys_, f=Constant(0), c_0=c_n, phi_0=porositys_n,
                                    bdries=solid_bdries, bcs=bcs_tracer, parameters=tracer_parameters)

            c_n.assign(c_)
            
            porositys_n.assign(porositys_)


            # Save output
            if(timestep % int(toutput/dt) == 0):

                logging.info("\n*** save output time %e s"%(current_time))
                logging.info("number of time steps %i"%timestep)
                logging.info("cycle # %i"%n_cycle)

                # may report Courant number or other important values that indicate how is doing the run

                c_n.rename("c", "tmp")
                c_out << (c_n, current_time)


if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Launch a simulation of PVS flow')

    # Add the arguments

    my_parser.add_argument('-j','--job_name',
                        type=str,
                        default="PVSandBrain",
                        help='Name of the job')

    my_parser.add_argument('-o','--output_folder',
                            type=str,
                            default="../output/",
                            help='Folder where the results are stored')

    my_parser.add_argument('-rv','--radius_vessel',
                        type=float,
                        default=8e-4,
                        help='Vessel radius as rest')

    my_parser.add_argument('-rpvs','--radius_pvs',
                        metavar='Rpvs',
                        type=float,
                        default=11e-4,
                        help='PVS outer radius as rest')

    my_parser.add_argument('-rbrain','--radius_brain',
                        type=float,
                        default=100e-4,
                        help='Outer radius of the brain domain')

    my_parser.add_argument('-l','--length',
                        type=float,
                        default=200e-4,
                        help='Length of the vessel')

    my_parser.add_argument('-ai',
                        type=float,
                        nargs='+',
                        default=[0.1],
                        help='List of ai')

    my_parser.add_argument('-fi',
                        type=float,
                        nargs='+',
                        default=[1],
                        help='List of fi')


    my_parser.add_argument('-phii',
                        type=float,
                        nargs='+',
                        default=[0],
                        help='List of phii')

    my_parser.add_argument('-tend',
                        type=float,
                        default=5,
                        help='final time')

    my_parser.add_argument('-toutput',
                        type=float,
                        default=5e-2,
                        help='output period for the adv-diff problem')        

    my_parser.add_argument('-toutputcycle',
                        type=float,
                        default=5e-2,
                        help='output period for the stoke biot solution on one period')               

    my_parser.add_argument('-dt','--time_step',
                        type=float,
                        default=1e-2,
                        help='time step')

    my_parser.add_argument('-mu','--viscosity',
                        type=float,
                        default=7e-3,
                        help='Dynamic viscosity of the fluid')

    my_parser.add_argument('-rho','--density',
                        type=float,
                        default=1,
                        help='Density of the fluid')

    my_parser.add_argument('-r','--resistance',
                        type=float,
                        default=-1,
                        help='Resistance at the inner side of the brain')

    my_parser.add_argument('-k','--ale_parameter',
                        type=float,
                        default=1,
                        help='ALE parameter')

    my_parser.add_argument('-nrpvs','--N_radial_fluid',
                        type=int,
                        default=8,
                        help='number of cells in the radial direction in the PVS')

    my_parser.add_argument('-nrbiot','--N_radial_biot',
                        type=int,
                        default=20,
                        help='number of cells in the radial direction in the biot domain')

    my_parser.add_argument('-s_biot','--biot_progression',
                        type=float,
                        default=2.5,
                        help='progression coefficient for the mesh in the biot domain')

    my_parser.add_argument('-nl','--N_axial',
                        type=int,
                        default=100,
                        help='number of cells in the axial direction')

    my_parser.add_argument('-d','--diffusion_coef',
                        type=float,
                        default=1e-7,
                        help='Molecular diffusion coefficient of the tracer')

    my_parser.add_argument('-s','--sigma',
                        type=float,
                        default=4e-4,
                        help='STD gaussian init for concentration')

    my_parser.add_argument('-xi','--initial_pos',
                        type=float,
                        default=100e-4,
                        help='Initial center of the gaussian for concentration')

    my_parser.add_argument('-perm','--biot_permeability',
                        type=float,
                        default=3.3e-13,
                        help='Permeability in the biot domain (cm2)')

    my_parser.add_argument('-E','--biot_youngmodulus',
                        type=float,
                        default=1000e3,
                        help='Young modulus in the Biot domain (dyn/cm2)')

    my_parser.add_argument('-nu','--biot_poisson',
                        type=float,
                        default=0.45,
                        help='Poisson coefficient in the Biot domain')

    my_parser.add_argument('-alpha','--biot_biotcoefficient',
                        type=float,
                        default=1,
                        help='Biot coefficient')

    my_parser.add_argument('-tau','--biot_tortuosity',
                        type=float,
                        default=0.5,
                        help='Tortuosity in the Biot domain')

    my_parser.add_argument('-phi','--biot_porosity',
                        type=float,
                        default=0.2,
                        help='Porosity in the Biot domain')     

    my_parser.add_argument('-mesh','--mesh_method',
                        type=str,
                        default='regular',
                        help='Method for the mesh construction')   
    args = my_parser.parse_args()


    # Execute the PVS simulation

    PVSbrain_simulation(args)


