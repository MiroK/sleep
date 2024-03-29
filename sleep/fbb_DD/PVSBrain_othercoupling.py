#! /usr/bin/env python3
import argparse
import logging
from datetime import datetime
import os

import numpy as np
from math import pi

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
    """ Solve the flow and tracer transport in the PVS (Stockes domain) and the brain (Biot domain) :
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


    logging.info('Vessel radius : %e cm'%Rv)
    logging.info('PVS radius : %e cm'%Rpvs)
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
    logging.info('N radial PVS: %e'%Nr)
    logging.info('N radial Biot: %e'%Nr_biot)
    logging.info('progression parameter in biot: %e'%s_biot)

    

    #time parameters
    logging.info('\n * Time')
    toutput=args.toutput
    tfinal=args.tend
    dt=args.time_step

    logging.info('final time: %e s'%tfinal)
    logging.info('output period : %e s'%toutput)
    logging.info('time step : %e s'%dt)

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
    logging.info('permeability : %e cm2'%conductivity)
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


        fluid_lines = [factory.addLine(*p) for p in ((a,b), (b, c), (c,d), (d,a))]
        named_lines = dict(zip(('bottom', 'pvs_right', 'interface', 'pvs_left'), fluid_lines))

        fluid_loop = factory.addCurveLoop(fluid_lines)
        fluid = factory.addPlaneSurface([fluid_loop])

        solid_lines = [factory.addLine(*p) for p in ((d,c), (c, e), (e,f), (f,d))]
        named_lines.update(dict(zip(('interface', 'brain_right', 'brain_top', 'brain_left'), solid_lines)))

        solid_loop = factory.addCurveLoop(solid_lines)
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

    fluid_coef = tracer_parameters.pop('%s_0' % coef)
    solid_coef = tracer_parameters.pop('%s_1' % coef)

    form = ((1/CellVolume(mesh))*fluid_coef*q*dx(0) +
                (1/CellVolume(mesh))*solid_coef*q*dx(1))
        
    tracer_parameters[coef] = Function(CoefSpace, assemble(form))


    # masks
    maskspace=FunctionSpace(mesh, 'DG', 0)
    q = TestFunction(maskspace)

    form = ((1/CellVolume(mesh))*Constant(0)*q*dx(0) +
                (1/CellVolume(mesh))*Constant(1)*q*dx(1))
                
    mask_solid=Function(CoefSpace, assemble(form))

    form = ((1/CellVolume(mesh))*Constant(1)*q*dx(0) +
                (1/CellVolume(mesh))*Constant(0)*q*dx(1))   

    mask_fluid=Function(maskspace, assemble(form))

    File(outputfolder+'fields'+'/mask_fluid.pvd') << mask_fluid
    File(outputfolder+'fields'+'/mask_solid.pvd') << mask_solid
    #FEM space

    logging.info(title1("Set FEM spaces"))

    logging.info('\n * Fluid')
    Vf_elm = VectorElement('Lagrange', triangle, 2)
    Qf_elm = FiniteElement('Lagrange', triangle, 1)
    Wf_elm = MixedElement([Vf_elm, Qf_elm])
    Wf = FunctionSpace(mesh_f, Wf_elm)
    logging.info('Velocity : "Lagrange", triangle, 2')
    logging.info('Pressure : "Lagrange", triangle, 1')

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
    Ct = FunctionSpace(mesh, Ct_elm)
    logging.info('Concentration : "Lagrange", triangle, 1')


    logging.info('\n * ALE')
    Va_elm = VectorElement('Lagrange', triangle, 1)
    Va = FunctionSpace(mesh_f, Va_elm)
    logging.info('ALE displacement: "Lagrange", triangle, 1')

    #Auxillary functions

    # Displacement in the solid domain
    # Fenics won't let us move mesh with quadratic displacement so
    Va_s = VectorFunctionSpace(mesh_s, 'CG', 1)

    # Displacement in the full domain
    Va_full = VectorFunctionSpace(mesh, 'CG', 1)

    # CHECK here
    # Should I take  VectorElement / Finite elements or VectorFunctionSpace/FunctionSpace with 'CG',1 , 2?
    FS_advvel= VectorFunctionSpace(mesh, 'CG', 2)
    #advvel_elt=VectorElement('Lagrange', triangle, 2)
    #FS_advvel = FunctionSpace(mesh, advvel_elt)


    # Porosity variables 
    FS_porosity= FunctionSpace(mesh, 'DG', 0)
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

    logging.info('\n * Cross section area parameters')
    ai=args.ai
    fi=args.fi
    phii=args.phii
    logging.info('ai (dimensionless): '+'%e '*len(ai)%tuple(ai))
    logging.info('fi (Hz) : '+'%e '*len(fi)%tuple(fi))
    logging.info('phii (rad) : '+'%e '*len(phii)%tuple(phii))


    # Vessel radius

    functionR = Rv*(1+sum([a*sin(2*pi*f*tn+phi) for a,f,phi in zip(ai,fi,phii)])) # displacement
    R_vessel = sympy.printing.ccode(functionR)

    functionV = sympy.diff(functionR,tn) # velocity
    V_vessel = sympy.printing.ccode(functionV)

    #Delta U for ALE at the vessel boundary
    functionUALE=Rv*(1+sum([a*sin(2*pi*f*tnp1+phi) for a,f,phi in zip(ai,fi,phii)]))- Rv*(1+sum([a*sin(2*pi*f*tn+phi) for a,f,phi in zip(ai,fi,phii)]))
    UALE_vessel = sympy.printing.ccode(functionUALE)   

    vf_bottom = Expression(('0',V_vessel ), tn = 0, degree=2)   # no slip no gap condition at vessel wall 
    uale_bottom = Expression(('0',UALE_vessel ), tn = 0, tnp1=1, degree=2) # displacement for ALE at vessel wall 


    driving_expressions = (vf_bottom,uale_bottom)


    # define functions for the interface BC

    # need CHECK : chose carefully the DG0, 1 ?
    tractionf_interface = Function(VectorFunctionSpace(mesh_f, 'CG', 1))  
    eta_interface = Function(VectorFunctionSpace(mesh_f, 'CG', 1))
    p_interface = Function(FunctionSpace(mesh_s, 'DG', 0))
    etas_interface = Function(VectorFunctionSpace(mesh_s, 'CG', 2))  
    aux = Function(VectorFunctionSpace(mesh_s, 'CG', 2))  # Auxiliary for u_s.np.np




    logging.info('\n * Lateral assumption')
    logging.info(lateral_bc)

    logging.info('\n * Fluid')
    logging.info('Left : zero pressure')

    if lateral_bc=='free' :
        logging.info('Right : zero pressure')
    elif lateral_bc=='resistance' :
        logging.info('Right : resistance')
    else :
        logging.info('Right : no flow')

    logging.info('Top : Interface condition')
    logging.info('Bottom : no slip no gap moving wall')


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
    

    logging.info('\n * ALE')
    logging.info('Left : no flux')
    logging.info('Right : no flux')
    logging.info('Top : no displacement')
    logging.info('Bottom : vessel displacement')

    # Now we wire up

    if lateral_bc=='free' :
        bcs_fluid = {'velocity': [(facet_lookup['F_bottom'],vf_bottom)],
                    'traction': [(facet_lookup['Interface'], tractionf_interface)],  
                    'pressure': [(facet_lookup['F_left'], Constant(0)),
                                (facet_lookup['F_right'], Constant(0))]}

    elif lateral_bc=='resistance' :

        Rpressure=Expression('R*Q+p0', R = resistance, Q=0, p0=0, degree=1) #
  
        # Compute pressure to impose according to the flow at previous time step and resistance.

        bcs_fluid = {'velocity': [(facet_lookup['F_bottom'],vf_bottom)],
                    'traction': [(facet_lookup['Interface'], tractionf_interface)],  
                    'pressure': [(facet_lookup['F_left'], Constant(0)),
                                 (facet_lookup['F_right'], Rpressure)]}
    else :
        bcs_fluid = {'velocity': [(facet_lookup['F_bottom'],vf_bottom),
                                (facet_lookup['F_right'], Constant((0,0)))], # I would like only normal flow to be zero 
                    'traction': [(facet_lookup['Interface'], tractionf_interface)],  
                    'pressure': [(facet_lookup['F_left'], Constant(0))]}     


    #todo : add case fluid free
    bcs_tracer = {'concentration': [(facet_lookup['F_left'], Constant(1)),
                                          (facet_lookup['S_left'], Constant(1))],
                    'flux': [(facet_lookup['F_right'], Constant(0)),
                            (facet_lookup['S_right'], Constant(0)),
                            (facet_lookup['F_bottom'], Constant(0)),
                            (facet_lookup['S_top'], Constant(0))]}

    bcs_solid = {   'elasticity': {
                        'displacement': [(facet_lookup['Interface'], etas_interface)],
                        'traction': [(facet_lookup['S_left'], Constant((0,0)))],
                        'displacement_x' : [(facet_lookup['S_right'],Constant((0)))],
                        'displacement_y' : [(facet_lookup['S_top'], Constant(0))]                         
                    },
                    'darcy': {
                        'pressure': [(facet_lookup['Interface'], p_interface),
                                    (facet_lookup['S_left'], Constant(0))],
                        'flux': [(facet_lookup['S_top'], Constant((0,0))),
                                (facet_lookup['S_right'], Constant((0,0)))]     
                    }
                }                   

    bcs_ale = {'dirichlet': [(facet_lookup['F_bottom'], uale_bottom),
                            (facet_lookup['Interface'], eta_interface)],
               'neumann': [(facet_lookup['F_left'], Constant((0, 0))),
                        (facet_lookup['F_right'], Constant((0, 0)))]}


    # Things for coupling
    # Todo cylindrical coordinates
    n_f, n_p = FacetNormal(mesh_f), FacetNormal(mesh_s)

    sigma_f = lambda u, p, mu=fluid_parameters['mu']: 2*mu*sym(grad(u)) - p*Identity(2)

    sigma_E = lambda eta, mu=solid_parameters['mu'], lmbda=solid_parameters['lmbda']: (
        2*mu*sym(grad(eta)) + lmbda*div(eta)*Identity(2)
    )
    sigma_p = lambda eta, p, alpha=solid_parameters['alpha']: sigma_E(eta)-alpha*p*Identity(2)

    interface_tag = facet_lookup['Interface']
    boundaries = (fluid_bdries, solid_bdries, interface_tag)


    # Initialisation : 


    logging.info(title1("Initialisation"))
    logging.info("\n * Fluid")
    logging.info("Velocity : zero field")
    logging.info("Pressure : zero field")
    uf_n = project(Constant((0, 0)), Wf.sub(0).collapse())
    pf_n =  project(Constant(0), Wf.sub(1).collapse())

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

    #cf_0 = Expression('x[1]<= Rpvs ? exp(-a*pow(x[0]-b, 2))*(x[1]-Rpvs)/(Rv-Rpvs) : 0 ', degree=2, a=1/2/sigma_gauss**2, b=xi_gauss, Rv=Rv, Rpvs=Rpvs)

   # cf_0 = Expression('x[0]<= 150e-4 ? 1 : 0 ', degree=2, a=1/2/sigma_gauss**2, b=xi_gauss, Rv=Rv, Rpvs=Rpvs)

    c_n =  project(Constant(0),Ct)#

    
 
    #class InitialCondition(UserExpression):
    #    def eval_cell(self, value, x, ufc_cell):
    #        if x[1] < Rpvs:
    #            value[0] = cf_0(x)
    #        else:
    #            value[0] = 0.0
    #c_n=Function(Ct)
    #c_n.interpolate(InitialCondition())



    porositys_n=project(Constant(porosity_0),FS_porositys)

    porosity_n=project(Constant(1)*mask_fluid+Constant(porosity_0)*mask_solid,FS_porosity)





    #Initial deformation of the fluid domain
    # We start at a time shift
    tshift=  1/4/fi[0] 

    # Update boundary conditions
    for expr in driving_expressions:
        hasattr(expr, 'tn') and setattr(expr, 'tn', 0)
        hasattr(expr, 'tnp1') and setattr(expr, 'tnp1', tshift)

    #Solve ALE and move mesh 
    # Transfert solid displacement at the interface to the fluid domain for ALE
    transfer_into(eta_interface, Constant((0,0)), boundaries)

    etaf_n = solve_ale(Va, f=Constant((0, 0)), bdries=fluid_bdries, bcs=bcs_ale,
                      parameters=ale_parameters)

    #CHECK is it okay to do that ? the extrapolated part is set to 0 with the mask function
    etaf_n.set_allow_extrapolation(True)

    #CHECK project or asign ?
    eta_n=project(mask_fluid*etaf_n,Va_full)

    ALE.move(mesh, eta_n)
    ALE.move(mesh_f, etaf_n)

    #it is needed for the mask function to work properly
    mesh.bounding_box_tree().build(mesh) 
    mesh_f.bounding_box_tree().build(mesh_f)


    # Save initial state
    uf_n.rename("vf", "tmp")
    pf_n.rename("pf", "tmp")

    uf_out << (uf_n, 0)
    pf_out << (pf_n, 0)
 

    etas_n.rename("us", "tmp")
    us_n.rename("qs", "tmp")
    ps_n.rename("ps", "tmp")

    etas_out << (etas_n, 0)
    qs_out << (us_n, 0)
    ps_out << (ps_n, 0)

    c_n.rename("c", "tmp")
    c_out << (c_n, 0)

    porosity_n.rename("phi", "tmp")
    phi_out << (porosity_n, 0)

    files=[csv_p,csv_u,csv_c]
    fields=[pf_n,uf_n.sub(0),c_n]
    
    slice_line = line([0,(Rpvs+Rv)/2],[L,(Rpvs+Rv)/2], 100)

    #todo add header for the cvs files

    #todo : do the cross section integrate here

    for csv_file,field in zip(files,fields) :
        #print the x scale
        values=np.linspace(0,L,100)
        row=[0]+list(values)
        csv_file.write(('%e'+', %e'*len(values)+'\n')%tuple(row))
        #print the initial 1D slice
        values = line_sample(slice_line, field) 
        row=[0]+list(values)
        csv_file.write(('%e'+', %e'*len(values)+'\n')%tuple(row))





    ############# RUN ############3

    logging.info(title1("Run"))

    # Time loop
    time = tshift
    timestep=0





    # Here I dont know if there will be several dt for advdiff and fluid solver
    while time < tfinal+tshift:

        time+=dt
        timestep+=1
        print('time', time-tshift)

        # Update boundary conditions
        for expr in driving_expressions:
            hasattr(expr, 'tn') and setattr(expr, 'tn', time)
            hasattr(expr, 'tnp1') and setattr(expr, 'tnp1', time+dt)

        if lateral_bc=='resistance' :
            z, r = SpatialCoordinate(mesh_f)
            ds = Measure('ds', domain=mesh_f, subdomain_data=fluid_bdries)
            n = FacetNormal(mesh_f)
            Flow=assemble(2*pi*r*dot(uf_n, n)*ds(facet_lookup['F_right']))

            print('Right outflow : %e \n'%Flow)
            setattr(Rpressure, 'Q', Flow)

        #Solve ALE and move mesh 
        # Transfert solid displacement at the interface to the fluid domain for ALE
        transfer_into(eta_interface, etas_n, boundaries)

        etaf_n = solve_ale(Va, f=Constant((0, 0)), bdries=fluid_bdries, bcs=bcs_ale,
                      parameters=ale_parameters)

        #CHECK is it okay to do that ? the extrapolated part is set to 0 with the mask function
        etas_n.set_allow_extrapolation(True)
        etaf_n.set_allow_extrapolation(True)


        #CHECK project or asign ?

        eta_n=project(mask_solid*etas_n+mask_fluid*etaf_n,Va_full)#etaf_n

        #This is quite long
        ALE.move(mesh, eta_n)
        ALE.move(mesh_f, etaf_n)
        ALE.move(mesh_s, interpolate(etas_n, Va_s))

        #is needed for the masks
        mesh.bounding_box_tree().build(mesh) 
        mesh_f.bounding_box_tree().build(mesh_f)
        mesh_s.bounding_box_tree().build(mesh_s)



        # Solve fluid problem        
        
        # Impose the velocity in the fluid at the interface, given dU/dt of the interface and the percolation velocity at last time step  
        #transfer_into(vf_interface, etas_n/Constant(dt) + us_n, boundaries)

        # Given that eta_0 and p_0 we compute traction sigma_f.n_f = -sigma_p.n_p
        transfer_into(tractionf_interface, -dot(sigma_p(etas_n, ps_n), n_p),  boundaries) 

        uf_, pf_ = solve_fluid(Wf, u_n=uf_n, p_n=pf_n,  f=Constant((0, 0)), bdries=fluid_bdries, bcs=bcs_fluid,
                            parameters=fluid_parameters)


        # Solve porous problem
        # Transfer the fluid pressure from fluid domain
        transfer_into(p_interface, pf_, boundaries)   

        # and normal stress from fluid domain
        # CHECK update n_f ? after ALE
        #transfer_into(traction_interface,-dot(sigma_f(uf_, pf_), n_f),boundaries) 

        # We set the displacement combining
        # uf.nf + (d/dt eta + u_s).np = 0 and uf.tau - d/dt eta. tau = 0 into
        # uf - d/dt eta - (u_s.nf)*nf = 0 i.e eta = dt*(uf - (u_s.nf)*nf) + eta_0 
        transfer_into(etas_interface,
                    Constant(dt)*uf_,
                    boundaries)  # eta = dt*uf

        transfer_into(aux,
                    Constant(dt)*dot(us_n, n_p)*n_p,
                    (solid_bdries, interface_tag))

        etas_interface.vector().axpy(-dt, aux.vector()) # eta = dt*uf - dt*u_s.np*np
        #etas_iface.vector().axpy(1.0, eta_s0.vector()) # eta += eta_0

        etas_, us_, ps_ = solve_solid(Ws, f1=Constant((0, 0)), f2=Constant(0), eta_0=etas_0, p_0=ps_n,
                                            bdries=solid_bdries, bcs=bcs_solid, parameters=solid_parameters)

        


        # Solve tracer problem
        # Todo : Maybe simplify and remove this as we are not using substeping ?
        tracer_parameters["T0"]=time

        # Compute the porosity in the Biot domain
        # Extend solid deformation to 3D : for cylindrical symmetry
        #deformation = as_vector((etas_[0], etas_[1],Constant(0)))
       
       
        #porositys_=project(porositys_n+solid_parameters['alpha']*div(etas_)+solid_parameters['s0']*(ps_-ps_n),FS_porositys)
        porositys_=project((porositys_n+div(etas_))/(1+div(etas_)),FS_porositys)
        porositys_.set_allow_extrapolation(True)
        porosity_=project(mask_solid*porositys_+mask_fluid*Constant(1),FS_porosity)

        uf_.set_allow_extrapolation(True)
        us_n.set_allow_extrapolation(True)

        #CHECK : I use us_n to be sure to have continuity at the interface. I dont know if this 'delayed' adv velocity in the solid domain is an issue.
        advection_velocity=project(mask_solid*us_n+mask_fluid*(uf_-etaf_n/Constant(dt)),FS_advvel)


        c_, T0= solve_adv_diff(Ct, velocity=advection_velocity, phi=porosity_, f=Constant(0), c_0=c_n, phi_0=porosity_n,
                                  bdries=full_bdries, bcs=bcs_tracer, parameters=tracer_parameters)


        # Update current solution
        uf_n.assign(uf_)
        pf_n.assign(pf_)

        etas_n.assign(etas_)
        us_n.assign(us_)
        ps_n.assign(ps_)

        c_n.assign(c_)

        porositys_n.assign(porositys_)
        porosity_n.assign(porosity_)


        # Save output
        if(timestep % int(toutput/dt) == 0):

            logging.info("\n*** save output time %e s"%(time-tshift))
            logging.info("number of time steps %i"%timestep)

            # may report Courant number or other important values that indicate how is doing the run

            uf_n.rename("vf", "tmp")
            pf_n.rename("pf", "tmp")
    
            uf_out << (uf_n, time-tshift)
            pf_out << (pf_n, time-tshift)
            etaf_n.rename("fluid_def", "tmp")
            File(outputfolder+'fields'+'/fluid_def.pvd') << (etaf_n, time-tshift)

            etas_n.rename("us", "tmp")
            us_n.rename("qs", "tmp")
            ps_n.rename("ps", "tmp")


            etas_out << (etas_n, time-tshift)
            qs_out << (us_n, time-tshift)
            ps_out << (ps_n, time-tshift)

            c_n.rename("c", "tmp")
            c_out << (c_n, time-tshift)

            porosity_n.rename("phi", "tmp")
            phi_out << (porosity_n, time-tshift)

            advection_velocity.rename("adv_vel", "tmp")
            File(outputfolder+'fields'+'/adv_vel.pvd') << (advection_velocity, time-tshift)

            eta_n.rename("mesh_def", "tmp")
            File(outputfolder+'fields'+'/mesh_def.pvd') << (eta_n, time-tshift)

            ## Get the 1 D profiles at umax (to be changed in cyl coordinate)
            #mesh_points=mesh_f.coordinates()                                                      
            #x=mesh_points[:,0]
            #y=mesh_points[:,1]
            #xmin=min(x)
            #xmax=max(x)
            #ymin=min(y)
            #ymax=max(y)
                        
            #slice_line = line([xmin,(ymin+ymax)/2],[xmax,(ymin+ymax)/2], 100)

            #logging.info('Rpvs : %e'%ymax)
            #logging.info('Rvn : %e'%ymin)

            #files=[csv_p,csv_u,csv_c]
            #fields=[pf_n,uf_n.sub(0),c_n]
            #field_names=['pressure (dyn/cm2)','axial velocity (cm/s)','concentration']

            ##todo : correct here the y integration (the interval changes with x due to deformations of the brain)
            ##need to CHECK : impose that the brain interface deform only along y ?
            #for csv_file,field,name in zip(files,fields,field_names) :
            #    #values = line_sample(slice_line, field)
            #    values =profile_cyl(field,xmin,xmax,ymin,ymax)
            #    logging.info('Max '+name+' : %.2e'%max(abs(values)))
            #    #logging.info('Norm '+name+' : %.2e'%field.vector().norm('linf'))
            #    row=[time-tshift]+list(values)
            #    csv_file.write(('%e'+', %e'*len(values)+'\n')%tuple(row))

            #csv_rv.write('%e, %e'%(time-tshift,ymin))




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
                        help='output period')                       

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
                        default=2e-8,
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
                        default=3e-13,
                        help='Permeability in the biot domain (cm2)')

    my_parser.add_argument('-E','--biot_youngmodulus',
                        type=float,
                        default=5000e3,
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
                        default=1,
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


