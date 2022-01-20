#! /usr/bin/env python3
import argparse
import logging
from datetime import datetime
import os

import numpy as np
from math import pi

from sleep.stages.cycles import State, Cycle
from sleep.fbb_DD.advection import solve_adv_diff_cyl as solve_adv_diff
from sleep.fbb_DD.fluid import solve_fluid_cyl as solve_fluid
from sleep.fbb_DD.ale import solve_ale_cyl as solve_ale
from sleep.utils import EmbeddedMesh
#from sleep.mesh import load_mesh2d
from dolfin import *
import sleep.fbb_DD.cylindrical as cyl

from sleep.stages.readConfig import ReadCycle



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

def slice_integrate(x,f,ymin,ymax,N=10) :
    vertical_line=line([x,ymin],[x,ymax], N)
    values=line_sample(vertical_line, f)
    r=np.linspace(ymin,ymax,N)
    integral=2*np.trapz(values,r)/(ymax-ymin) #cylindrical coordinate
    return integral

def profile(f,xmin,xmax,ymin,ymax,Nx=100,Ny=10):
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



def PVS_simulation(args):
    """ Solve the flow and tracer transport in the PVS :
    Outputs :
    - a logfile with information about the simulation
    - .pvd files with the u, p and c field at specified args.toutput time period
    
    Return : u, p, c 1D array of the u, p, c fields on the middle line """



    # output folder name
    outputfolder=args.output_folder+'/'+args.job_name+'/'

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)


    #if not os.path.exists(outputfolder+'profiles'):
    #    os.makedirs(outputfolder+'profiles')

    if not os.path.exists(outputfolder+'fields'):
        os.makedirs(outputfolder+'fields')

    # Create output files

    #txt files
    csv_p=open(args.output_folder+'/'+args.job_name+'_pressure.txt', 'w')
    csv_u=open(args.output_folder+'/'+args.job_name+'_velocity.txt', 'w')
    csv_c=open(args.output_folder+'/'+args.job_name+'_concentration.txt', 'w')
    csv_rv=open(args.output_folder+'/'+args.job_name+'_radius.txt', 'w')

    csv_mass=open(args.output_folder+'/'+args.job_name+'_mass.txt', 'w')

    #pvd files
    uf_out, pf_out= File(outputfolder+'fields'+'/uf.pvd'), File(outputfolder+'fields'+'/pf.pvd')
    c_out= File(outputfolder+'fields'+'/c.pvd')
    facets_out=File(outputfolder+'fields'+'/facets.pvd')

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # log to a file
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = args.output_folder+'/'+args.job_name+'_PVSinfo.log'
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

    logging.info(title1("Simulation of the PVS flow and tracer transport using non steady solver and diffusion-advection solvers"))

    logging.info("Date and time:"+datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

    logging.info('Job name : '+args.job_name)

    logging.debug('logging initialized')


    # Set parameters

    logging.info(title1("Parameters"))

    # Geometry params
    logging.info('\n * Geometry')
    Rv = args.radius_vessel # centimeters
    Rpvs =args.radius_pvs # centimeters
    L = args.length # centimeters

    logging.info('Vessel radius : %e cm'%Rv)
    logging.info('PVS radius : %e cm'%Rpvs)
    logging.info('PVS length : %e cm'%L)

    #Mesh
    logging.info('\n * Mesh')
    #number of cells in the radial direction
    Nr=args.N_radial
    DR=(Rpvs-Rv)/Nr
    #number of cells in the axial direction
    if args.N_axial :
        Nl=args.N_axial
    else :
        Nl=round(L/DR)

    DY=L/Nl
    
    logging.info('N axial : %i'%Nl)
    logging.info('N radial : %e'%Nr)

    

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



    logging.info('\n * ALE')
    kappa=args.ale_parameter
    logging.info('ALE parameter: %e '%kappa)

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
    tracer_parameters={'kappa': D, 'dt':dt_advdiff}
    ale_parameters = {'kappa': kappa}



    # Setup of boundary conditions
    logging.info(title1("Boundary conditions"))


    logging.info('\n * Cross section area parameters')

    if args.cycle:
        logging.info('frequency and amplitude data from cycle '+args.cycle)
        timedependentfa=True
    else :
        timedependentfa=False
        ai=args.ai
        fi=args.fi
        phii=args.phii
        logging.info('ai (dimensionless): '+'%e '*len(ai)%tuple(ai))
        logging.info('fi (Hz) : '+'%e '*len(fi)%tuple(fi))
        logging.info('phii (rad) : '+'%e '*len(phii)%tuple(phii))

    if timedependentfa:

        ##
        logging.info('creation of cycle')


        cycleObj=ReadCycle('../stages/cycles.yml',args.cycle)
        totalcycletime=np.sum(cycleObj.durations)
        spantime,listspana,listspanf,spanRv,spanh0,spanRpvs=cycleObj.generatedata(int(tfinal/totalcycletime)+1)

        logging.info('*** Simulation of %s cycle '%cycleObj.name)


        # adjust last time in order to be able to interpolate
        #spantime[-1]=max(tfinal+2*dt,spantime[-1])

        from scipy.interpolate import interp1d
        varflist={}
        varalist={}

        for freq in listspanf:
            varflist[freq]=interp1d(spantime,listspanf[freq])
            varalist[freq]=interp1d(spantime,listspana[freq])



        varh0=interp1d(spantime,spanh0)
        varRpvs=interp1d(spantime,spanRpvs)


        #varda=interp1d(spantime,dadt,kind="previous")

        fs=1/dt  # test if same result if multiplying by 10 ?

        time=0 + np.arange(int((tfinal+2*dt)*fs))/fs # longer than the simulation time because we need tn+1 for the U ALE computation



        modulation={}
        for freq in listspanf:
            modulation[freq]=varalist[freq](time)*np.sin(2*np.pi*np.cumsum(varflist[freq](time))/fs)

        OuterRadius=varRpvs(time)
        InnerRadius=varRpvs(time) -varh0(time)*(1+modulation['cardiac']+modulation['resp']+modulation['LF']+modulation['VLF'])

        # define the thickness interpolation function 
        interph0=interp1d(time,varh0(time)*(1+modulation['cardiac']+modulation['resp']+modulation['LF']+modulation['VLF']))
       
        ## More easy here to take the numerical derivative of the radius
        ##dadt= np.array(list(np.diff(vara(time))/np.diff(time))+[0.0])
        ##dRadiusdt= -(Rpvs-Rv)*(dadt*np.sin(2*np.pi*np.cumsum(varf(time))/fs)+vara(time)*np.cos((2*np.pi*np.cumsum(varf(time))/fs))*(2*np.pi*varf(time)))
        douterRadiusdt= np.array(list(np.diff(OuterRadius)/np.diff(time))+[0.0])
        dinnerRadiusdt= np.array(list(np.diff(InnerRadius)/np.diff(time))+[0.0])

        # define an expression for the radius and the derivative

        class Interp(UserExpression):
            def __init__(self, x, y, **kwargs):
                super().__init__(self, **kwargs)
                self.interp =interp1d(x,y)
                self.tn=0

            def eval(self, values, x):
                values[0] = 0
                values[1] = self.interp(self.tn)

            def value_shape(self):
                return (2,)

        class Interpdiff(UserExpression):
            def __init__(self, x, y, **kwargs):
                super().__init__(self, **kwargs)
                self.interp =interp1d(x,y)
                self.tn=0
                self.tnp1=dt

            def eval(self, values, x):
                values[0] = 0
                values[1] = self.interp(self.tnp1)-self.interp(self.tn)

            def value_shape(self):
                return (2,)               

        interpRpvs=Interp( time,OuterRadius, degree=1)
        interpRv=Interp( time,InnerRadius, degree=1)
        
        #initial values for Rv and Rpvs
        uale_top = Interpdiff( time, OuterRadius,degree=1)
        uale_bottom = Interpdiff( time, InnerRadius,degree=1)
        vf_top = Interp( time,douterRadiusdt, degree=1)
        vf_bottom = Interp( time,dinnerRadiusdt, degree=1)

        Rvfunction=interp1d(time, InnerRadius)
        Rpvsfunction=interp1d(time, OuterRadius)
        dRvdtfunction=interp1d(time,dinnerRadiusdt)

    else :

        import sympy
        tn = sympy.symbols("tn")
        tnp1 = sympy.symbols("tnp1")
        sin = sympy.sin
        sqrt = sympy.sqrt

        functionR = Rpvs -(Rpvs-Rv)*(1+sum([a*sin(2*pi*f*tn+phi) for a,f,phi in zip(ai,fi,phii)])) # displacement
        R_vessel = sympy.printing.ccode(functionR)

        functionV = sympy.diff(functionR,tn) # velocity
        V_vessel = sympy.printing.ccode(functionV)

        #Delta U for ALE. I dont really like this
        functionUALE=-(Rpvs-Rv)*(1+sum([a*sin(2*pi*f*tnp1+phi) for a,f,phi in zip(ai,fi,phii)]))+(Rpvs-Rv)*(1+sum([a*sin(2*pi*f*tn+phi) for a,f,phi in zip(ai,fi,phii)]))
        UALE_vessel = sympy.printing.ccode(functionUALE)   

        vf_bottom = Expression(('0',V_vessel ), tn = 0, degree=2)   # no slip no gap condition at vessel wall 
        uale_bottom = Expression(('0',UALE_vessel ), tn = 0, tnp1=1, degree=2) # displacement for ALE at vessel wall 

        vf_top = Constant((0,0 ))
        uale_top = Constant((0,0 )) 

        ## Is there a better way to do that ?
        Rvfunction=lambda t : functionR.subs(tn,t).evalf()
        Rpvsfunction=lambda t : Rpvs
        dRvdtfunction=lambda t : functionV.subs(tn,t).evalf()

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

    logging.info('Top : no slip no gap fixed wall')
    logging.info('Bottom : no slip no gap moving wall')

    logging.info('\n * Tracer concentration')

    sas_bc=args.sasbc
    init_concentration_type=args.c0init
    init_concentration_value=args.c0valuePVS


    logging.info('Left BC scenario :',sas_bc)

    c_SAS= Expression('m/VCSF', m=0, VCSF=40e-3, degree=2)



    #initial concentration in SAS
    if sas_bc=='scenarioA':
        cSAS=0
    else :
        cSAS=args.c0valueSAS

    # number of vessels used for mass balance
    Nvessels=6090
    # initial volume of CSF in PVS
    VPVS=((np.pi*Rpvsfunction(0)**2)-(np.pi*Rvfunction(0)**2))*L*Nvessels

    # initial volume of CSF in SAS : assumed to be 10 times larger than volume in PVS
    VCSF=VPVS*10 #40e-3 

    # initial pressure of the CSF
    PCSF=4 # mmHg
    # initial volume of arterial blood
    Vblood=4e-3 # ml
    # equivalent vessel length used for the compliance function and assessement of ICP
    leq=Vblood/(np.pi*Rvfunction(0)**2)

    # initial tracer mass in the CSF
    m=cSAS*VCSF

    # constant production of CSF
    Qprod=6e-6  # ml/s
    
    # Outflow resistance
    Rcsf=5/1.7e-5 # mmHg/(ml/s)
    # CSF compliance
    Ccsf=1e-3 #ml/mmHg
    
    


    if sas_bc=='scenarioA' :
        logging.info('Left : zero concentration')
        # initial outflow of CSF (not used, just for output file)
        Qout=0
    elif sas_bc=='scenarioB' :
        logging.info('Left : mass conservation, no CSF outflow')
        # initial outflow of CSF
        Qout=0
    elif sas_bc=='scenarioC' :
        logging.info('Left : mass conservation, constant CSF outflow')
        # initial outflow of CSF
        Qout=Qprod
    elif sas_bc=='scenarioD' :
        logging.info('Left : mass conservation, pressure dependent CSF outflow')
        # initial outflow of CSF
        Qout=Qprod
        # venous pressure
        Pss=PCSF-Qout*Rcsf


    if lateral_bc=='free' :
        logging.info('Right : zero concentration')
    else :
        logging.info('Right : no flux')


    logging.info('Top : no flux')
    logging.info('Bottom : no flux')

    logging.info('\n * ALE')
    logging.info('Left : no flux')
    logging.info('Right : no flux')
    logging.info('Top : no displacement')
    logging.info('Bottom : vessel displacement')




    # Mesh
    logging.info(title1('Meshing'))

    logging.info('cell size : %e cm'%(np.sqrt(DR**2+DY**2)))
    logging.info('nb cells: %i'%(Nl*Nr*2))

    mesh_f= RectangleMesh(Point(0, Rvfunction(0)), Point(L, Rpvsfunction(0)), Nl, Nr)

    ## Refinement at the SAS boundary
    if args.refineleft :
        x = mesh_f.coordinates()[:,0]
        y = mesh_f.coordinates()[:,1]

        #Deformation of the mesh

        def deform_mesh(x, y):
            x=L*(x/L)**2.5
            return [x, y]

        x_bar, y_bar = deform_mesh(x, y)
        xy_bar_coor = np.array([x_bar, y_bar]).transpose()
        mesh_f.coordinates()[:] = xy_bar_coor
        mesh_f.bounding_box_tree().build(mesh_f)

    fluid_bdries = MeshFunction("size_t", mesh_f, mesh_f.topology().dim()-1,0)

    # Label facets
    xy = mesh_f.coordinates().copy()              
    x, y = xy.T

    xmin=x.min()
    xmax=x.max()
    ymin=y.min()
    ymax=y.max()


    tol=min(DR,DY)/2 #cm

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


    bbottom.mark(fluid_bdries,  2) 
    btop.mark(fluid_bdries,  4)  
    bleft.mark(fluid_bdries, 1) 
    bright.mark(fluid_bdries,  3) 

    

    facet_lookup = {'x_min': 1 ,'y_min': 2, 'x_max': 3, 'y_max': 4}

    facets_out << fluid_bdries




    # Now we wire up

    if lateral_bc=='free' :
        bcs_fluid = {'velocity': [(facet_lookup['y_min'],vf_bottom),
                                (facet_lookup['y_max'], vf_top)],
                    'traction': [],  
                    'pressure': [(facet_lookup['x_min'], Constant(0)),
                                (facet_lookup['x_max'], Constant(0))]}

    elif lateral_bc=='resistance' :

        Rpressure=Expression('R*Q+p0', R = resistance, Q=0, p0=0, degree=1) #
  
        # Compute pressure to impose according to the flow at previous time step and resistance.

        bcs_fluid = {'velocity': [(facet_lookup['y_min'],vf_bottom),
                                (facet_lookup['y_max'], vf_top)],
                    'traction': [],  
                    'pressure': [(facet_lookup['x_min'], Constant(0)),
                                 (facet_lookup['x_max'], Rpressure)]}
    else :
        bcs_fluid = {'velocity': [(facet_lookup['y_min'],vf_bottom),
                                (facet_lookup['y_max'], vf_top),
                                (facet_lookup['x_max'], Constant((0,0)))], # I would like only normal flow to be zero 
                    'traction': [],  
                    'pressure': [(facet_lookup['x_min'], Constant(0))]}     


    bcs_tracer = {'concentration': [(facet_lookup['x_min'], c_SAS)],
                    'flux': [(facet_lookup['x_max'], Constant(0)),
                            (facet_lookup['y_max'], Constant(0)),
                            (facet_lookup['y_min'], Constant(0))]}

    if lateral_bc=='free' :
        bcs_tracer = {'concentration': [(facet_lookup['x_max'], Constant(0)),
                                        (facet_lookup['x_min'], c_SAS)],
                    'flux': [(facet_lookup['y_max'], Constant(0)),
                            (facet_lookup['y_min'], Constant(0))]}




    bcs_ale = {'dirichlet': [(facet_lookup['y_min'], uale_bottom),
                            (facet_lookup['y_max'], uale_top)],
               'neumann': [(facet_lookup['x_min'], Constant((0, 0))),
                        (facet_lookup['x_max'], Constant((0, 0)))]}




    # We collect the time dependent BC for update
    driving_expressions = (uale_bottom,vf_bottom,uale_top,vf_top)




    #FEM space

    logging.info(title1("Set FEM spaces"))

    logging.info('\n * Fluid')
    Vf_elm = VectorElement('Lagrange', triangle, 2)
    Qf_elm = FiniteElement('Lagrange', triangle, 1)
    Wf_elm = MixedElement([Vf_elm, Qf_elm])
    Wf = FunctionSpace(mesh_f, Wf_elm)
    logging.info('Velocity : "Lagrange", triangle, 2')
    logging.info('Pressure : "Lagrange", triangle, 1')

    logging.info('\n * Tracer')
    Ct_elm = FiniteElement('Lagrange', triangle, 1)
    Ct = FunctionSpace(mesh_f, Ct_elm)
    logging.info('Concentration : "Lagrange", triangle, 1')


    logging.info('\n * ALE')
    Va_elm = VectorElement('Lagrange', triangle, 1)
    Va = FunctionSpace(mesh_f, Va_elm)
    logging.info('ALE displacement: "Lagrange", triangle, 1')
    


    # Initialisation : 
    logging.info(title1("Initialisation"))
    logging.info("\n * Fluid")
    logging.info("Velocity : zero field")
    logging.info("Pressure : zero field")
    uf_n = project(Constant((0, 0)), Wf.sub(0).collapse())
    pf_n =  project(Constant(0), Wf.sub(1).collapse())

    logging.info("\n * Tracer")



    if init_concentration_type=='gaussian' :
        logging.info("Concentration : Gaussian profile")
        logging.info("                Centered at xi = %e"%xi_gauss)
        logging.info("                STD parameter = %e"%sigma_gauss)
        logging.info("                Max value=%e"%init_concentration_value)

        c_0 = Expression('c0*exp(-a*pow(x[0]-b, 2)) ', degree=1, a=1/2/sigma_gauss**2, b=xi_gauss,c0=init_concentration_value)
        c_n =  project(c_0,Ct)
    elif init_concentration_type=='constant' :
        logging.info("Concentration : Uniform profile")
        logging.info("Value=%e"%init_concentration_value)

        ## Initialisation 
        c_n =  project(Constant(init_concentration_value),Ct)
    elif init_concentration_type=='null' :
        logging.info("Concentration : zero in the vessel")


        ## Initialisation 
        c_n =  project(Constant(0),Ct)
    else :
        logging.info("Concentration : Uniform profile (default)")
        logging.info("Value=%e"%0)
        ## Initialisation 
        c_n =  project(Constant(0),Ct)

    

    # Save initial state
    uf_n.rename("uf", "tmp")
    pf_n.rename("pf", "tmp")
    c_n.rename("c", "tmp")
    uf_out << (uf_n, 0)
    pf_out << (pf_n, 0)
    c_out << (c_n, 0)

    files=[csv_p,csv_u,csv_c]
    fields=[pf_n,uf_n.sub(0),c_n]
    
    slice_line = line([0,(Rpvs+Rv)/2],[L,(Rpvs+Rv)/2], 100)

    for csv_file,field in zip(files,fields) :
        #print the x scale
        values=np.linspace(0,L,100)
        row=[0]+list(values)
        csv_file.write(('%e'+', %e'*len(values)+'\n')%tuple(row))
        #print the initial 1D slice
        values = line_sample(slice_line, field) 
        row=[0]+list(values)
        csv_file.write(('%e'+', %e'*len(values)+'\n')%tuple(row))



    ############# RUN ############

    logging.info(title1("Run"))

    # Time loop
    time = 0.
    timestep=0

    z, r = SpatialCoordinate(mesh_f)
    ds = Measure('ds', domain=mesh_f, subdomain_data=fluid_bdries)
    n = FacetNormal(mesh_f)

    # volume of pvs
    volume = 2*np.pi*assemble(Constant(1.0)*r*dx(mesh_f))
    # integral of concentration
    intc = 2*np.pi*assemble(r*c_n*dx(mesh_f))


    # tracer mass out of the system
    mout=0

    csv_mass.write('%s, %s, %s, %s, %s, %s, %s, %s, %s\n'%('time', 'mass PVS', 'mass CSF', 'mass out', 'Total mass','PVS volume', 'CSF volume', 'P csf', 'Q out'))
    csv_mass.write('%e, %e, %e, %e, %e, %e, %e, %e, %e\n'%(time,Nvessels*intc,m,mout,Nvessels*intc+m+mout, Nvessels*volume, VCSF,PCSF,Qout))

    
    ### ALE deformation function
    expressionDeformation=Expression(("0","(x[1]-rmax)/(rmax-rmin)*h+Rpvs-x[1]"), rmin=0, rmax=1 ,Rpvs=1,h=1, degree=1)

    
    # Extend normal to 3d as GradAxisym(scalar) is 3-vector
    normal = as_vector((Constant(-1),
                        Constant(0),
                        Constant(0)))    

    # Here I dont know if there will be several dt for advdiff and fluid solver
    while time < tfinal:

        for expr in driving_expressions:
            hasattr(expr, 'tn') and setattr(expr, 'tn', time)
            hasattr(expr, 'tnp1') and setattr(expr, 'tnp1', time+dt)

        

        if lateral_bc=='resistance' :
            Flow=assemble(2*pi*r*dot(uf_n, n)*ds(facet_lookup['x_max']))

            setattr(Rpressure, 'Q', Flow)

        #Solve ALE and move mesh 
        #eta_f = solve_ale(Va, f=Constant((0, 0)), bdries=fluid_bdries, bcs=bcs_ale, parameters=ale_parameters)


        # Or just compute the deformation
        xy=mesh_f.coordinates()
        x, y = xy.T

        expressionDeformation.Rpvs=Rpvsfunction(time)      
        expressionDeformation.h=Rpvsfunction(time)-Rvfunction(time)  
        expressionDeformation.rmin=min(y)
        expressionDeformation.rmax=max(y)   

        #eta_f = interpolate(expressionDeformation,VectorFunctionSpace(mesh_f,"CG",1))
        eta_f=project(expressionDeformation,Va)


        ALE.move(mesh_f, eta_f)
        mesh_f.bounding_box_tree().build(mesh_f)


        # update the coordinates
        z, r = SpatialCoordinate(mesh_f)
        ds = Measure('ds', domain=mesh_f, subdomain_data=fluid_bdries)
        n = FacetNormal(mesh_f)



        # Solve fluid problem
        uf_, pf_ = solve_fluid(Wf, u_0=uf_n,  f=Constant((0, 0)), bdries=fluid_bdries, bcs=bcs_fluid,
                            parameters=fluid_parameters)


        # Solve tracer problem
        tracer_parameters["T0"]=time
        tracer_parameters["nsteps"]=1
        tracer_parameters["dt"]=dt


        # If the fluid is exiting the PVS we compute the amount of mass entering the SAS. The tracer left BC is free.
        # If the fluid is entering the PVS then we impose the concentration in the SAS at the left BC.

        #Fluid flow at the BC
        FluidFlow=assemble(2*pi*r*dot(uf_, n)*ds(facet_lookup['x_min']))

        #  n is directed in the outward direction

        if FluidFlow>0 : 

            bcs_tracer = {'concentration': [],
                        'flux': [(facet_lookup['x_min'], Constant(0)),
                                (facet_lookup['x_max'], Constant(0)),
                                (facet_lookup['y_max'], Constant(0)),
                                (facet_lookup['y_min'], Constant(0))]}
        else :
            cmean=assemble(2*pi*r*c_n*ds(facet_lookup['x_min']))/assemble(2*pi*r*Constant(1)*ds(facet_lookup['x_min']))
            # we allow the possibility to use a relaxation here
            alpha=0. # 0 means no relaxation
            c_imposed=(1-alpha)*cSAS+alpha*cmean

            bcs_tracer = {'concentration': [(facet_lookup['x_min'], Constant(c_imposed))],
                        'flux': [(facet_lookup['x_max'], Constant(0)),
                                (facet_lookup['y_max'], Constant(0)),
                                (facet_lookup['y_min'], Constant(0))]}


        c_, T0= solve_adv_diff(Ct, velocity=uf_-eta_f/Constant(dt), phi=Constant(1), f=Constant(0), c_0=c_n, phi_0=Constant(1),
                            bdries=fluid_bdries, bcs=bcs_tracer, parameters=tracer_parameters)


        Massflow=assemble(2*pi*r*dot(uf_-eta_f/Constant(dt), n)*c_*ds(facet_lookup['x_min']))
        Massdiffusion=tracer_parameters["kappa"]*assemble(2*pi*r*dot(cyl.GradAxisym(c_), normal)*ds(facet_lookup['x_min']))


        if sas_bc=='scenarioD' :

            # update CSF outflow
            Qout = max((PCSF-Pss)/Rcsf,0) # valve
            # update CSF pressure
            PCSF += dt/Ccsf*(Qprod-Qout)+ np.pi*leq*(Rvfunction(time+dt)**2-Rvfunction(time)**2)/Ccsf
                        


        if sas_bc=='scenarioA' :
            if FluidFlow>0 :     
                # mainly advection
                mout+=dt*Nvessels*Massflow

            # lost mass in the PVS due to diffusion
            mout+=-dt*Nvessels*Massdiffusion
                

        else:
            # Advected mass
            m+=dt*Nvessels*Massflow-dt*Qout*cSAS
            # Adding diffusion
            m+=-dt*Nvessels*Massdiffusion

            mout+=Qout*cSAS*dt


        # update tracer concentration in SAS
        cSAS=m/VCSF



        # Update current solution
        uf_n.assign(uf_)
        pf_n.assign(pf_)
        c_n.assign(c_)

        #Update time
        time += dt
        timestep+=1

        # Save output
        if(timestep % int(toutput/dt) == 0):
         
            logging.info("\n*** save output time %e s"%time)
            logging.info("number of time steps %i"%timestep)

            # may report Courant number or other important values that indicate how is doing the run

            uf_.rename("uf", "tmp")
            pf_.rename("pf", "tmp")
            c_.rename("c", "tmp")
            uf_out << (uf_, time)
            pf_out << (pf_, time)
            c_out << (c_, time)


            # Get the 1 D profiles at umax (to be changed in cyl coordinate)
            mesh_points=mesh_f.coordinates()                                                      
            x=mesh_points[:,0]
            y=mesh_points[:,1]
            xmin=min(x)
            xmax=max(x)
            ymin=min(y)
            ymax=max(y)
                        
            #slice_line = line([xmin,(ymin+ymax)/2],[xmax,(ymin+ymax)/2], 100)

            logging.info('Rpvs : %e'%ymax)
            logging.info('Rvn : %e'%ymin)

            files=[csv_p,csv_u,csv_c]
            fields=[pf_n,uf_n.sub(0),c_n]
            field_names=['pressure (dyn/cm2)','axial velocity (cm/s)','concentration']

            for csv_file,field,name in zip(files,fields,field_names) :
                #values = line_sample(slice_line, field)
                values =profile(field,xmin,xmax,ymin,ymax)
                logging.info('Max '+name+' : %.2e'%max(abs(values)))
                #logging.info('Norm '+name+' : %.2e'%field.vector().norm('linf'))
                row=[time]+list(values)
                csv_file.write(('%e'+', %e'*len(values)+'\n')%tuple(row))
                csv_file.flush()

            csv_rv.write(('%e, %e\n')%(time,ymin))
            csv_rv.flush()

            # volume of pvs
            volume = 2*np.pi*assemble(Constant(1.0)*r*dx(mesh_f))
            # integral of concentration
            intc = 2*np.pi*assemble(r*c_*dx(mesh_f))

            csv_mass.write('%e, %e, %e, %e, %e, %e, %e, %e, %e\n'%(time,Nvessels*intc,m,mout,Nvessels*intc+m+mout, Nvessels*volume, VCSF,PCSF,Qout))
            csv_mass.flush()
            




if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Launch a simulation of PVS flow')

    # Add the arguments

    my_parser.add_argument('-j','--job_name',
                        type=str,
                        default="PVS",
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
                        default=10e-4,
                        help='PVS outer radius as rest')

    my_parser.add_argument('-lpvs','--length',
                        type=float,
                        default=100e-4,
                        help='Length of the vessel')

    my_parser.add_argument('-ai',
                        type=float,
                        nargs='+',
                        default=[0.01],
                        help='List of ai')

    my_parser.add_argument('-fi',
                        type=float,
                        nargs='+',
                        default=[10],
                        help='List of fi')


    my_parser.add_argument('-phii',
                        type=float,
                        nargs='+',
                        default=[0],
                        help='List of phii')

    my_parser.add_argument('-tend',
                        type=float,
                        default=1,
                        help='final time')

    my_parser.add_argument('-toutput',
                        type=float,
                        default=0.01,
                        help='output period')                       

    my_parser.add_argument('-dt','--time_step',
                        type=float,
                        default=1e-3,
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
                        default=0,
                        help='Resistance at the inner side of the brain')

    my_parser.add_argument('-k','--ale_parameter',
                        type=float,
                        default=1,
                        help='ALE parameter')

    my_parser.add_argument('-nr','--N_radial',
                        type=int,
                        default=8,
                        help='number of cells in the radial direction')

    my_parser.add_argument('-nl','--N_axial',
                        type=int,
                        default=0,
                        help='number of cells in the axial direction')

    my_parser.add_argument('-d','--diffusion_coef',
                        type=float,
                        default=2e-8,
                        help='Diffusion coefficient of the tracer')

    my_parser.add_argument('-s','--sigma',
                        type=float,
                        default=1e-4,
                        help='STD gaussian init for concentration')

    my_parser.add_argument('-xi','--initial_pos',
                        type=float,
                        default=0,
                        help='Initial center of the gaussian for concentration')
 
    my_parser.add_argument('-c0init',
                        type=str,
                        default='',
                        help='Type of initialisation for the concentration')

    my_parser.add_argument('-sasbc',
                        type=str,
                        default='scenarioA',
                        help='Choice of the scenario for the left concentration boundary condition')

    my_parser.add_argument('-c0valueSAS',
                        type=float,
                        default=0,
                        help='Initial value of the concentration in the SAS')

    my_parser.add_argument('-c0valuePVS',
                        type=float,
                        default=1,
                        help='Initial value of the concentration in the PVS')

    my_parser.add_argument('-cycle',
                        type=str,
                        default='',
                        help='cycle name, must be defined in the cycles.yml config file')
    
    my_parser.add_argument('-refineleft',
                        type=bool,
                        default=False,
                        help='Refine the mesh on the left side')

    args = my_parser.parse_args()


    # Execute the PVS simulation

    PVS_simulation(args)


# python3 PVS_cyclessimulation.py -j REMsleepnew -lpvs 200e-4 -c0init constant -c0value 50 -sasbc scenarioA -cycle REMsleep -tend 420 -toutput 1 -dt 2e-2 -r -1 -nr 4 -nl 50 -d 2e-7
# python3 PVS_cyclessimulation.py -j REMsleepin -lpvs 200e-4 -c0init null -c0value 50 -sasbc scenarioB -cycle REMsleep -tend 420 -toutput 1 -dt 2e-2 -r -1 -nr 4 -nl 50 -d 2e-7 -refineleft True

#python3 PVS_cyclessimulation.py -j testSAS -lpvs 200e-4 -c0init constant -c0value 50 -sasbc scenarioD -fi 10 -ai 0.01 -tend 1 -toutput 1e-2 -dt 1e-2 -r -1 -nr 6 -nl 100 -d 2e-7 
