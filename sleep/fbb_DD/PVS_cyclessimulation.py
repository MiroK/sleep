#! /usr/bin/env python3
import argparse
#from asyncio.sslproto import constants
import logging
from datetime import datetime
import os
from ssl import ALERT_DESCRIPTION_UNRECOGNIZED_NAME

import numpy as np
from math import pi

from src.fem.advection import solve_adv_diff_cyl as solve_adv_diff
from src.fem.fluid import solve_fluid_cyl as solve_fluid
from src.fem.utils import EmbeddedMesh

#from sleep.mesh import load_mesh2d
from dolfin import *

import src.fem.cylindrical as cyl



# Define line function for 1D slice evaluation
def line_sample(line, f, fill=np.nan):
    values = fill*np.ones(len(line))
    for i, xi in enumerate(line):
        try:
            values[i] = f(xi)
        except RuntimeError:
            continue
    return values


def slice_integrate_cyl(x, f, ymin, ymax, N=10):
    vertical_line = line([x, ymin], [x, ymax], N)
    values = line_sample(vertical_line, f)
    r = np.linspace(ymin, ymax, N)
    integral = 2*np.trapz(values*r, r)/(ymax**2-ymin **
                                        2)  # cylindrical coordinate
    return integral


def profile_cyl(f, xmin, xmax, ymin, ymax, Nx=100, Ny=10):
    spanx = np.linspace(xmin, xmax, Nx)
    values = [slice_integrate_cyl(x, f, ymin, ymax, N=Ny) for x in spanx]
    return np.array(values)


def slice_integrate(x, f, ymin, ymax, N=10):
    vertical_line = line([x, ymin], [x, ymax], N)
    values = line_sample(vertical_line, f)
    r = np.linspace(ymin, ymax, N)
    integral = 2*np.trapz(values, r)/(ymax-ymin)  # cylindrical coordinate
    return integral


def profile(f, xmin, xmax, ymin, ymax, Nx=100, Ny=10):
    spanx = np.linspace(xmin, xmax, Nx)
    values = [slice_integrate_cyl(x, f, ymin, ymax, N=Ny) for x in spanx]
    return np.array(values)


def line(A, B, nsteps):
    A = np.array(A)
    B = np.array(B)
    return A + (B-A)*np.linspace(0, 1, nsteps).reshape((-1, 1))


def title1(string):
    line1 = '\n'+'*'*100+'\n'
    line2 = '**   '+string+'\n'
    line3 = '*'*100
    return line1 + line2 + line3


def PVS_simulation(args):
    """ Solve the flow and tracer transport in the PVS :
    Outputs :
    - a logfile with information about the simulation
    - .pvd files with the u, p and c field at specified args.toutput time period

    Return : u, p, c 1D array of the u, p, c fields on the middle line """

    # output folder name
    outputfolder = args.output_folder+'/'+args.job_name+'/'

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    # if not os.path.exists(outputfolder+'profiles'):
    #    os.makedirs(outputfolder+'profiles')

    if not os.path.exists(outputfolder+'fields'):
        os.makedirs(outputfolder+'fields')

    # Create output files

    # txt files
    csv_p = open(args.output_folder+'/'+args.job_name+'_pressure.txt', 'w')
    csv_u = open(args.output_folder+'/'+args.job_name+'_velocity.txt', 'w')
    csv_c = open(args.output_folder+'/'+args.job_name +
                 '_concentration.txt', 'w')
    csv_rv = open(args.output_folder+'/'+args.job_name+'_radius.txt', 'w')

    csv_mass = open(args.output_folder+'/'+args.job_name+'_mass.txt', 'w')

    # pvd files
    uf_out, pf_out = File(outputfolder+'fields' +
                          '/uf.pvd'), File(outputfolder+'fields'+'/pf.pvd')
    c_out = File(outputfolder+'fields'+'/c.pvd')
    facets_out = File(outputfolder+'fields'+'/facets.pvd')

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # log to a file
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = args.output_folder+'/'+args.job_name+'_PVSinfo.log'
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setLevel(logging.INFO)
    #formatter = logging.Formatter("%(asctime)s %(filename)s, %(lineno)d, %(funcName)s: %(message)s")
    # file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # log to the console
    console_handler = logging.StreamHandler()
    level = logging.INFO
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # initialise logging

    # initialise logging

    logging.info(
        ' ______     ______        _                 _       _   _             ')
    logging.info(
        '|  _ \ \   / / ___|   ___(_)_ __ ___  _   _| | __ _| |_(_) ___  _ __  ')
    logging.info(
        "| |_) \ \ / /\___ \  / __| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \ ")
    logging.info(
        '|  __/ \ V /  ___) | \__ \ | | | | | | |_| | | (_| | |_| | (_) | | | |')
    logging.info(
        '|_|     \_/  |____/  |___/_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_|\n')

    logging.info(title1(
        "Simulation of the PVS flow and tracer transport using stoke solver and diffusion-advection solver"))

    logging.info("Date and time:" +
                 datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

    logging.info('Job name : '+args.job_name)

    logging.debug('logging initialized')

    # Set parameters

    logging.info(title1("Parameters"))

    # Geometry params
    logging.info('\n * Geometry')
    Rv = args.radius_vessel  # centimeters
    Rpvs = args.radius_pvs  # centimeters
    aSMC=args.alpha_SMC # ratio
    L = args.length  # centimeters

    

    logging.info('Vessel radius : %e cm' % Rv)
    logging.info('PVS radius : %e cm' % Rpvs)
    logging.info('PVS length : %e cm' % L)
    logging.info('SMC initial area coverage : %e pc' % aSMC)

    # test presence of the SAS compartment on the mesh
    isSAS = args.issas

    if isSAS:
        logging.info('Add a SAS compartment on the left')
        Rsas = args.radius_sas+Rv
        Lsas = args.length_sas
        logging.info('SAS length : %e cm' % Lsas)
        logging.info('SAS radius : %e cm' % Rsas)


    # Mesh
    logging.info('\n * Mesh')
    # number of cells in the radial direction
    Nr = args.N_radial
    DR = (Rpvs-Rv)/Nr
    # number of cells in the axial direction
    if args.N_axial:
        Nl = args.N_axial
    else:
        Nl = round(L/DR)

    DY = L/Nl

    logging.info('N axial : %i' % Nl)
    logging.info('N radial : %e' % Nr)

    # time parameters
    logging.info('\n * Time')
    toutput = args.toutput
    tfinal = args.tend
    dt = args.time_step

    logging.info('final time: %e s' % tfinal)
    logging.info('output period : %e s' % toutput)
    logging.info('time step : %e s' % dt)

    # approximate CFL for fluid solver : need to compute max velocity depending on the wall displacement...
    # maybe just add a warning in computation with actual velocity
    # Uapprox=500e-4 #upper limit for extected max velocity
    # CFL_dt=0.25*DY/Uapprox
    # if  CFL_dt < dt :
    #    logging.warning('The specified time step of %.2e s does not fullfil the fluid CFL condition. New fluid time step : %.2e s'%(dt, CFL_dt))
    # dt_fluid=min(dt,CFL_dt)
    dt_fluid = dt

    # approximate CFL for tracer solver
    dt_advdiff = dt

    # material parameters
    logging.info('\n * Fluid properties')
    mu = args.viscosity
    rho = args.density

    logging.info('density: %e g/cm3' % rho)
    logging.info('dynamic viscosity : %e dyn s/cm2' % mu)

    logging.info('\n* Tracer properties')
    D = args.diffusion_coef
    sigma_gauss = args.sigma
    logging.info('Free diffusion coef: %e cm2/s' % D)
    logging.info('STD of initial gaussian profile: %e ' % sigma_gauss)
    xi_gauss = args.initial_pos
    logging.info('Initial position: %e cm2' % xi_gauss)

    # oscillation parameters
    ai = args.ai
    fi = args.fi
    phii = args.phii
    logging.info('ai (dimensionless): '+'%e '*len(ai) % tuple(ai))
    logging.info('fi (Hz) : '+'%e '*len(fi) % tuple(fi))
    logging.info('phii (rad) : '+'%e '*len(phii) % tuple(phii))

    if aSMC >= (1-max(ai)) :
        logging.info('The area of SMC is larger than the deformation. Please chose a smaller value.')
        logging.info('We change the SMC area to be half of free space after deformation.')
        aSMC=(1-max(ai))/2
        logging.info('new SMC initial area coverage : %e pc' % aSMC)

    logging.info('\n * Lateral BC')
    resistance = args.resistance
    logging.info('inner resistance: %e ' % resistance)
    if resistance == 0:
        lateral_bc = 'free'
        logging.info('right BC will be set to the free assumption')
    elif resistance < 0:
        lateral_bc = 'noflow'
        logging.info('right BC will be set to the no flow assumption')
    else:
        lateral_bc = 'resistance'
        logging.info('right BC will be set to the resistance assumption')

    fluid_parameters = {'mu': mu, 'rho': rho, 'dt': dt_fluid}
    tracer_parameters = {'kappa': D, 'dt': dt_advdiff}


    # Setup of boundary conditions
    logging.info(title1("Boundary conditions"))

    logging.info('\n * Cross section area parameters')

    import sympy
    tn = sympy.symbols("tn")
    tnp1 = sympy.symbols("tnp1")
    sin = sympy.sin
    cos = sympy.cos
    sqrt = sympy.sqrt
                                  

    # ai is the change of area
    functionR = sqrt(Rpvs**2 -(Rpvs**2-Rv**2)*(1-sum([a*cos(2*pi*f*tn+phi) for a,f,phi in zip(ai,fi,phii)]))+aSMC*(Rpvs**2-Rv**2)) # displacement
    functionUALE=sqrt(Rpvs**2 -(Rpvs**2-Rv**2)*(1-sum([a*cos(2*pi*f*tnp1+phi) for a,f,phi in zip(ai,fi,phii)]))+aSMC*(Rpvs**2-Rv**2))- sqrt(Rpvs**2 -(Rpvs**2-Rv**2)*(1-sum([a*cos(2*pi*f*tn+phi) for a,f,phi in zip(ai,fi,phii)]))+aSMC*(Rpvs**2-Rv**2)) 
        

    functionV = sympy.diff(functionR, tn)  # velocity
    V_vessel = sympy.printing.ccode(functionV)

    UALE_vessel = sympy.printing.ccode(functionUALE)

    # no slip no gap condition at vessel wall
    vf_bottom = Expression(('0', V_vessel), tn=0, degree=2)
    # displacement for ALE at vessel wall
    uale_bottom = Expression(('0', UALE_vessel), tn=0, tnp1=1, degree=2)

    vf_top = Constant((0, 0))
    uale_top = Constant((0, 0))

    def Rvfunction(t): return functionR.subs(tn, t).evalf()
    def Rpvsfunction(t): return Rpvs
    def dRvdtfunction(t): return functionV.subs(tn, t).evalf()

    if isSAS:
        import sympy
        tn = sympy.symbols("tn")
        sin = sympy.sin
        # We add the possibility for rigid moion of the brain
        a_rigid = args.arigid  # cm
        f_rigid = args.frigid  # Hz
        logging.info(
            'ridig motion of the brain, amplitude : %.2e um' % a_rigid)
        logging.info(
            'ridig motion of the brain, frequency : %.2e Hz' % f_rigid)

        functionY = a_rigid*sin(2*pi*f_rigid*tn)
        functionVbone = sympy.diff(functionY, tn)  # velocity
        V_bone = sympy.printing.ccode(functionVbone)

        vf_bone = Expression(('0', V_bone), tn=0, degree=2)

    logging.info('\n * Lateral assumption')
    logging.info(lateral_bc)

    logging.info('\n * Fluid')
    logging.info('Left : zero pressure')

    if lateral_bc == 'free':
        logging.info('Right : zero pressure')
    elif lateral_bc == 'resistance':
        logging.info('Right : resistance')
    else:
        logging.info('Right : no flow')

    logging.info('Top : no slip no gap fixed wall')
    logging.info('Bottom : no slip no gap moving wall')

    logging.info('\n * Tracer concentration')

    sas_bc = args.sasbc
    init_concentration_type = args.c0init
    init_concentration_value = args.c0valuePVS

    logging.info('Left BC scenario :', sas_bc)

    if lateral_bc == 'free':
        logging.info('Right : zero concentration')
        productionrate = 0
    else:
        productionrate = args.productionrate
        if productionrate:
            logging.info(
                'Right : imposed solute production rate : %e ([c]/s)' % args.productionrate)
        else:
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

    if isSAS:
        Rv = Rvfunction(0)
        Rpvs = Rpvsfunction(0)
        DR = (Rpvs-Rv)/Nr

        logging.info('cell size : %e cm' % (DR))

        from sleep.mesh import mesh_model2d, load_mesh2d, set_mesh_size
        import gmsh

        gmsh.initialize(['', '-format', 'msh2'])

        model = gmsh.model

        import math
        Apvs0 = math.pi*Rpvs**2
        Av0 = math.pi*Rv**2
        A0 = Apvs0-Av0

        # progressive mesh
        factory = model.occ
        a = factory.addPoint(-Lsas, Rv, 0)
        b = factory.addPoint(L, Rv, 0)
        c = factory.addPoint(L, Rpvs, 0)
        d = factory.addPoint(0, Rpvs, 0)
        e = factory.addPoint(0, Rsas, 0)
        f = factory.addPoint(-Lsas, Rsas, 0)

        fluid_lines = [factory.addLine(
            *p) for p in ((a, b), (b, c), (c, d), (d, e), (e, f), (f, a))]
        named_lines = dict(zip(('bottom', 'pvs_right', 'pvs_top',
                           'brain_surf', 'sas_top', 'sas_left'), fluid_lines))

        fluid_loop = factory.addCurveLoop(fluid_lines)
        fluid = factory.addPlaneSurface([fluid_loop])

        factory.synchronize()

        model.addPhysicalGroup(2, [fluid], 1)

        for name in named_lines:
            tag = named_lines[name]
            model.addPhysicalGroup(1, [tag], tag)

        # boxes for mesh refinement

        boxes = []
        # add box on the PVS for mesh
        field = model.mesh.field
        fid = 1
        field.add('Box', fid)
        field.setNumber(fid, 'XMin', 0)
        field.setNumber(fid, 'XMax', L)
        field.setNumber(fid, 'YMin', Rvfunction(0))
        field.setNumber(fid, 'YMax', Rpvsfunction(0))
        field.setNumber(fid, 'VIn', DR)
        field.setNumber(fid, 'VOut', DR*50)
        field.setNumber(fid, 'Thickness', Rsas)

        boxes.append(fid)

        # Combine
        field.add('Min', 2)
        field.setNumbers(2, 'FieldsList', boxes)
        field.setAsBackgroundMesh(2)

        model.occ.synchronize()

        h5_filename = outputfolder+'/mesh.h5'
        tags = {'cell': {'F': 1},
                'facet': {}}
        mesh_model2d(model, tags, h5_filename)

        mesh_f, markers, lookup = load_mesh2d(h5_filename)

        gmsh.finalize()

    else:
        Rv = Rvfunction(0)
        Rpvs = Rpvsfunction(0)
        # simple PVS mesh
        logging.info('cell size : %e cm' % (np.sqrt(DR**2+DY**2)))
        logging.info('nb cells: %i' % (Nl*Nr*2))

        mesh_f = RectangleMesh(Point(0, Rv),
                               Point(L, Rpvs), Nl, Nr)

        # Refinement at the SAS boundary
        if args.refineleft:
            x = mesh_f.coordinates()[:, 0]
            y = mesh_f.coordinates()[:, 1]

            # Deformation of the mesh

            def deform_mesh(x, y):
                x = L*(x/L)**2.5
                return [x, y]

            x_bar, y_bar = deform_mesh(x, y)
            xy_bar_coor = np.array([x_bar, y_bar]).transpose()
            mesh_f.coordinates()[:] = xy_bar_coor
            mesh_f.bounding_box_tree().build(mesh_f)

    fluid_bdries = MeshFunction("size_t", mesh_f, mesh_f.topology().dim()-1, 0)

    # Label facets
    xy = mesh_f.coordinates().copy()
    x, y = xy.T

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    tol = min(DR, DY)/2  # cm

    if isSAS:

        class Boundary_sas_left(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], -Lsas, tol)

        # downstream
        class Boundary_pvs_right(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], L, tol)

        class Boundary_sas_top(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], Rsas, tol) and (x[0] < tol)

        class Boundary_pvs_top(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], Rpvs, tol) and (x[0] > -tol)

        # brain
        class Boundary_brainsurf(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], 0, tol) and (x[1] > Rpvs-tol)

        class Boundary_bottom(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], ymin, tol)

        btop_sas = Boundary_sas_top()
        btop_pvs = Boundary_pvs_top()
        bbottom = Boundary_bottom()

        bvert_left = Boundary_sas_left()
        bvert_brain = Boundary_brainsurf()
        bvert_right = Boundary_pvs_right()

        btop_sas.mark(fluid_bdries,  1)
        btop_pvs.mark(fluid_bdries,  2)
        bbottom.mark(fluid_bdries,  3)

        bvert_left.mark(fluid_bdries,  4)
        bvert_brain.mark(fluid_bdries,  5)
        bvert_right.mark(fluid_bdries,  6)

        facet_lookup = {'sas_out': 1, 'pvs_tissue': 2, 'vessel': 3,
                        'sas_bone': 4, 'sas_tissue': 5, 'pvs_end': 6}

    else:
        # simple PVS mesh

        class Boundary_left(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], xmin, tol)  # left

        class Boundary_right(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], xmax, tol)  # right

        class Boundary_bottom(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], ymin, tol)  # bottom

        class Boundary_top(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], ymax, tol)  # top

        btop = Boundary_top()
        bbottom = Boundary_bottom()
        bleft = Boundary_left()
        bright = Boundary_right()

        bbottom.mark(fluid_bdries,  2)
        btop.mark(fluid_bdries,  4)
        bleft.mark(fluid_bdries, 1)
        bright.mark(fluid_bdries,  3)

        facet_lookup = {'sas_out': 1, 'vessel': 2,
                        'pvs_end': 3, 'pvs_tissue': 4}

    facets_out << fluid_bdries

    # Now we wire up

    rate_prod = Expression(
        'rate/surface', rate=productionrate, surface=1, degree=1)

    if lateral_bc == 'free':
        bcs_fluid = {'velocity': [(facet_lookup['vessel'], vf_bottom),
                                  (facet_lookup['pvs_tissue'], vf_top)],
                     'traction': [],
                     'pressure': [(facet_lookup['sas_out'], Constant(0)),
                                  (facet_lookup['pvs_end'], Constant(0))]}

    elif lateral_bc == 'resistance':

        Rpressure = Expression('R*Q+p0', R=resistance, Q=0, p0=0, degree=1)

        # Compute pressure to impose according to the flow at previous time step and resistance.

        bcs_fluid = {'velocity': [(facet_lookup['vessel'], vf_bottom),
                                  (facet_lookup['pvs_tissue'], vf_top)],
                     'traction': [],
                     'pressure': [(facet_lookup['sas_out'], Constant(0)),
                                  (facet_lookup['pvs_end'], Rpressure)]}
    else:
        bcs_fluid = {'velocity': [(facet_lookup['vessel'], vf_bottom),
                                  (facet_lookup['pvs_tissue'], vf_top),
                                  (facet_lookup['pvs_end'], Constant((0, 0)))],  # I would like only normal flow to be zero
                     'traction': [],
                     'pressure': [(facet_lookup['sas_out'], Constant(0))]}

    # This is overwritten later depending on the scenario

    bcs_tracer_in = {'concentration': [(facet_lookup['sas_out'], 0)],
                     'flux': [(facet_lookup['pvs_end'], rate_prod),
                              (facet_lookup['pvs_tissue'], Constant(0)),
                              (facet_lookup['vessel'], Constant(0))]}

    bcs_tracer_out = {'concentration': [],
                      'flux': [(facet_lookup['sas_out'], Constant(0)),
                               (facet_lookup['pvs_end'], rate_prod),
                               (facet_lookup['pvs_tissue'], Constant(0)),
                               (facet_lookup['vessel'], Constant(0))]}

    # todo : add possibility to have other BC at pvs_end
    # if lateral_bc=='free' :
    #    bcs_tracer = {'concentration': [(facet_lookup['pvs_end'], Constant(0)),
    #                                    (facet_lookup['sas_out'], c_SAS)],
    #                'flux': [(facet_lookup['pvs_tissue'], Constant(0)),
    #                        (facet_lookup['vessel'], Constant(0))]}

    # add BC on the extra boundary in the mesh of the SAS
    if isSAS:
        bcs_fluid['velocity'].append((facet_lookup['sas_bone'], vf_bone))
        bcs_fluid['velocity'].append(
            (facet_lookup['sas_tissue'], Constant((0, 0))))

        bcs_tracer_in['flux'].append((facet_lookup['sas_bone'], Constant(0)))
        bcs_tracer_in['flux'].append((facet_lookup['sas_tissue'], Constant(0)))
        bcs_tracer_out['flux'].append((facet_lookup['sas_bone'], Constant(0)))
        bcs_tracer_out['flux'].append(
            (facet_lookup['sas_tissue'], Constant(0)))



    # We collect the time dependent BC for update
    driving_expressions = [uale_bottom, vf_bottom, uale_top, vf_top]

    if isSAS:
        driving_expressions.append(vf_bone)

    # FEM space

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

    c_SAS = Expression('m/VCSF', m=0, VCSF=40e-3, degree=2)

    # initial concentration in SAS
    if sas_bc == 'scenarioA':
        cSAS = 0
    else:
        cSAS = args.c0valueSAS

    # number of vessels used for mass balance
    Nvessels = 6090
    # initial volume of CSF in PVS
    z, r = SpatialCoordinate(mesh_f)
    ds = Measure('ds', domain=mesh_f, subdomain_data=fluid_bdries)
    n = FacetNormal(mesh_f)

    # volume of pvs
    VPVS = 2*np.pi*assemble(Constant(1.0)*r*dx(mesh_f))

    # initial volume of CSF in SAS : assumed to be 10 times larger than volume in PVS
    VCSF = 10*VPVS  # 40e-3

    # initial pressure of the CSF
    PCSF = 4  # mmHg
    # initial volume of arterial blood
    Vblood = 4e-3  # ml
    # equivalent vessel length used for the compliance function and assessement of ICP
    leq = Vblood/(np.pi*Rvfunction(0)**2)

    # initial tracer mass in the CSF
    m = cSAS*VCSF

    # constant production of CSF
    Qprod = 6e-6  # ml/s

    # Outflow resistance
    Rcsf = 5/1.7e-5  # mmHg/(ml/s)
    # CSF compliance
    Ccsf = 1e-3  # ml/mmHg

    if sas_bc == 'scenarioA':
        logging.info('Left : zero concentration')
        # initial outflow of CSF (not used, just for output file)
        Qout = 0
    elif sas_bc == 'scenarioB':
        logging.info('Left : mass conservation, no CSF outflow')
        # initial outflow of CSF
        Qout = 0
    elif sas_bc == 'scenarioC':
        logging.info('Left : mass conservation, constant CSF outflow')
        # initial outflow of CSF
        Qout = Qprod
    elif sas_bc == 'scenarioD':
        logging.info(
            'Left : mass conservation, pressure dependent CSF outflow')
        # initial outflow of CSF
        Qout = Qprod
        # venous pressure
        Pss = PCSF-Qout*Rcsf
    if sas_bc == 'scenarioE':
        logging.info('Left : constant concentration')
        Qout = 0

    logging.info("\n * Fluid")
    logging.info("Velocity : zero field")
    logging.info("Pressure : zero field")
    uf_n = project(Constant((0, 0)), Wf.sub(0).collapse())
    pf_n = project(Constant(0), Wf.sub(1).collapse())

    logging.info("\n * Tracer")

    if init_concentration_type == 'gaussian':
        logging.info("Concentration : Gaussian profile")
        logging.info("                Centered at xi = %e" % xi_gauss)
        logging.info("                STD parameter = %e" % sigma_gauss)
        logging.info("                Max value=%e" % init_concentration_value)

        c_0 = Expression('c0*exp(-a*pow(x[0]-b, 2)) ', degree=1, a=1 /
                         2/sigma_gauss**2, b=xi_gauss, c0=init_concentration_value)
        c_n = project(c_0, Ct)
    elif init_concentration_type == 'uniform':
        logging.info("Concentration : Uniform profile")

        if isSAS:
            logging.info("Value in PVS=%e" % init_concentration_value)
            logging.info("Value in SAS=%e" % cSAS)
            c_0 = Expression('x[0]>0 ? cPVS : cSAS ', degree=2,
                             cPVS=init_concentration_value, cSAS=cSAS)
            c_n = project(c_0, Ct)
        else:
            logging.info("Value=%e" % init_concentration_value)
            c_n = project(Constant(init_concentration_value), Ct)
    else:
        logging.info("Concentration : zero in the PVS (default)")
        c_n = project(Constant(0), Ct)

    # Save initial state
    uf_n.rename("uf", "tmp")
    pf_n.rename("pf", "tmp")
    c_n.rename("c", "tmp")
    uf_out << (uf_n, 0)
    pf_out << (pf_n, 0)
    c_out << (c_n, 0)

    files = [csv_p, csv_u, csv_c]
    fields = [pf_n, uf_n.sub(0), c_n]

    slice_line = line([0, (Rpvs+Rv)/2], [L, (Rpvs+Rv)/2], 100)

    for csv_file, field in zip(files, fields):
        # print the x scale
        values = np.linspace(0, L, 100)
        row = [0]+list(values)
        csv_file.write(('%e'+', %e'*len(values)+'\n') % tuple(row))
        # print the initial 1D slice
        values = line_sample(slice_line, field)
        row = [0]+list(values)
        csv_file.write(('%e'+', %e'*len(values)+'\n') % tuple(row))

    ############# RUN ############

    logging.info(title1("Run"))

    # Time loop
    time = 0.
    timestep = 0

    z, r = SpatialCoordinate(mesh_f)
    ds = Measure('ds', domain=mesh_f, subdomain_data=fluid_bdries)
    n = FacetNormal(mesh_f)

    # volume of pvs
    volume = 2*np.pi*assemble(Constant(1.0)*r*dx(mesh_f))
    # integral of concentration
    intc = 2*np.pi*assemble(r*c_n*dx(mesh_f))

    # tracer mass out of the system
    mout = 0

    csv_mass.write('%s, %s, %s, %s, %s, %s, %s, %s, %s\n' % ('time', 'mass PVS', 'mass CSF',
                   'mass out', 'Total mass', 'PVS volume', 'CSF volume', 'P csf', 'Q out'))
    csv_mass.write('%e, %e, %e, %e, %e, %e, %e, %e, %e\n' % (
        time, Nvessels*intc, m, mout, Nvessels*intc+m+mout, Nvessels*volume, VCSF, PCSF, Qout))

    # ALE deformation function
    expressionDeformation = Expression(
        ("0", "x[1]<=rpvs ? (x[1]-rpvs)/(rpvs-rvessel)*htarget+rpvstarget-x[1]:rpvstarget-rpvs"), rvessel=0, rpvs=1, rpvstarget=1, htarget=1, degree=1)

    # Extend normal to 3d as GradAxisym(scalar) is 3-vector
    normal = as_vector((Constant(-1),
                        Constant(0),
                        Constant(0)))

    # Here I dont know if there will be several dt for advdiff and fluid solver
    while time < tfinal:

        for expr in driving_expressions:
            hasattr(expr, 'tn') and setattr(expr, 'tn', time)
            hasattr(expr, 'tnp1') and setattr(expr, 'tnp1', time+dt)

        if lateral_bc == 'resistance':
            Flow = assemble(2*pi*r*dot(uf_n, n)*ds(facet_lookup['pvs_end']))

            setattr(Rpressure, 'Q', Flow)

        # compute the deformation of the mesh
        xy = mesh_f.coordinates()
        x, y = xy.T

        expressionDeformation.rpvstarget = Rpvsfunction(time)
        expressionDeformation.htarget = Rpvsfunction(time)-Rvfunction(time)
        # We look only in the PVS (x>0) not the SAS
        expressionDeformation.rvessel = min(y[x > 0])
        expressionDeformation.rpvs = max(y[x > 0])

        #eta_f = interpolate(expressionDeformation,VectorFunctionSpace(mesh_f,"CG",1))
        eta_f = project(expressionDeformation, Va)

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
        tracer_parameters["T0"] = time
        tracer_parameters["nsteps"] = 1
        tracer_parameters["dt"] = dt

        # If the fluid is exiting the PVS we compute the amount of mass entering the SAS. The tracer left BC is free.
        # If the fluid is entering the PVS then we impose the concentration in the SAS at the left BC.

        # Fluid flow at the BC
        FluidFlow = assemble(2*pi*r*dot(uf_, n)*ds(facet_lookup['sas_out']))

        #  n is directed in the outward direction : is it ?

        print('fluid flow : ', FluidFlow)

        if FluidFlow > 0:
            # then the fluid is going out and we impose natural BC for concentration
            bcs_tracer = bcs_tracer_out
        else:
            # then the fluid is going in and we impose the SAS concentration
            cmean = assemble(2*pi*r*c_n*ds(facet_lookup['sas_out']))/assemble(
                2*pi*r*Constant(1)*ds(facet_lookup['sas_out']))
            # we allow the possibility to use a relaxation here
            alpha = 0.  # 0 means no relaxation
            c_imposed = (1-alpha)*cSAS+alpha*cmean
            c_imposed = max(c_imposed, 0)

            bcs_tracer = bcs_tracer_in
            bcs_tracer['concentration'] = [
                (facet_lookup['sas_out'], Constant(c_imposed))]

        c_, T0 = solve_adv_diff(Ct, velocity=uf_-eta_f/Constant(dt), phi=Constant(1), f=Constant(0), c_0=c_n, phi_0=Constant(1),
                                bdries=fluid_bdries, bcs=bcs_tracer, parameters=tracer_parameters)

        Massflow = assemble(2*pi*r*dot(uf_-eta_f/Constant(dt), n)
                            * c_*ds(facet_lookup['sas_out']))
        Massdiffusion = tracer_parameters["kappa"]*assemble(
            2*pi*r*dot(cyl.GradAxisym(c_), normal)*ds(facet_lookup['sas_out']))

        if sas_bc == 'scenarioD':

            # update CSF outflow
            Qout = max((PCSF-Pss)/Rcsf, 0)  # valve
            # update CSF pressure
            PCSF += dt/Ccsf*(Qprod-Qout) + np.pi*leq * \
                (Rvfunction(time+dt)**2-Rvfunction(time)**2)/Ccsf

            # link between leq and Nvessels ?

        if sas_bc == 'scenarioA':
            if FluidFlow >= 0:
                # mainly advection
                mout += dt*Nvessels*Massflow
                # lost mass in the PVS due to diffusion
                mout += -dt*Nvessels*Massdiffusion

        else:
            # Advected mass
            m += dt*Nvessels*Massflow-dt*Qout*cSAS
            if FluidFlow >= 0:  # when in-flow we impose c sas at the boundary so no c gradient
                # Adding diffusion
                m += -dt*Nvessels*Massdiffusion

            mout += Qout*cSAS*dt

        # update the volume of CSF
        # VCSF+=dt*Nvessels*FluidFlow
        # should correspond to the volume change due to vessel dilation
        # VCSF+=np.pi*leq*(Rvfunction(time+dt)**2-Rvfunction(time)**2)

        # update tracer concentration in SAS
        if sas_bc == 'scenarioE':
            # we do not update the concentration
            cSAS = cSAS
        else:
            cSAS = m/VCSF

        rate_prod.surface = assemble(2*pi*r*ds(facet_lookup['pvs_end']))

        # Update current solution
        uf_n.assign(uf_)
        pf_n.assign(pf_)
        c_n.assign(c_)

        # Update time
        time += dt
        timestep += 1

        # Save output
        if(timestep % int(toutput/dt) == 0):

            logging.info("\n*** save output time %e s" % time)
            logging.info("number of time steps %i" % timestep)

            # may report Courant number or other important values that indicate how is doing the run

            uf_.rename("uf", "tmp")
            pf_.rename("pf", "tmp")
            c_.rename("c", "tmp")
            uf_out << (uf_, time)
            pf_out << (pf_, time)
            c_out << (c_, time)

            # Get the 1 D profiles at umax (to be changed in cyl coordinate)
            mesh_points = mesh_f.coordinates()
            x = mesh_points[:, 0]
            y = mesh_points[:, 1]
            xmin = min(x)
            xmax = max(x)
            ymin = min(y[x > 0])
            ymax = max(y[x > 0])

            # update the coordinates
            z, r = SpatialCoordinate(mesh_f)
            ds = Measure('ds', domain=mesh_f, subdomain_data=fluid_bdries)
            n = FacetNormal(mesh_f)

            #slice_line = line([xmin,(ymin+ymax)/2],[xmax,(ymin+ymax)/2], 100)

            logging.info('Rpvs : %e' % ymax)
            logging.info('Rvn : %e' % ymin)

            files = [csv_p, csv_u, csv_c]
            fields = [pf_n, uf_n.sub(0), c_n]
            field_names = ['pressure (dyn/cm2)',
                           'axial velocity (cm/s)', 'concentration']

            for csv_file, field, name in zip(files, fields, field_names):
                #values = line_sample(slice_line, field)
                values = profile(field, xmin, xmax, ymin, ymax)
                logging.info('Max '+name+' : %.2e' % max(abs(values)))
                #logging.info('Norm '+name+' : %.2e'%field.vector().norm('linf'))
                row = [time]+list(values)
                csv_file.write(('%e'+', %e'*len(values)+'\n') % tuple(row))
                csv_file.flush()

            csv_rv.write(('%e, %e\n') % (time, ymin))
            csv_rv.flush()

            # volume of pvs
            volume = 2*np.pi*assemble(Constant(1.0)*r*dx(mesh_f))
            # integral of concentration
            intc = 2*np.pi*assemble(r*c_*dx(mesh_f))

            csv_mass.write('%e, %e, %e, %e, %e, %e, %e, %e, %e\n' % (
                time, Nvessels*intc, m, mout, Nvessels*intc+m+mout, Nvessels*volume, VCSF, PCSF, Qout))
            csv_mass.flush()


if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(
        description='Launch a simulation of PVS flow')

    # Add the arguments

    my_parser.add_argument('-j', '--job_name',
                           type=str,
                           default="PVS",
                           help='Name of the job')

    my_parser.add_argument('-o', '--output_folder',
                           type=str,
                           default="../output/",
                           help='Folder where the results are stored')

    my_parser.add_argument('-rv', '--radius_vessel',
                           type=float,
                           default=8e-4,
                           help='Vessel radius as rest')

    my_parser.add_argument('-rpvs', '--radius_pvs',
                           metavar='Rpvs',
                           type=float,
                           default=10e-4,
                           help='PVS outer radius as rest')

    my_parser.add_argument('-aSMC', '--alpha_SMC',
                           metavar='aSMC',
                           type=float,
                           default=0,
                           help='pourcentage of the initial area occupied by smooth muscle cells (to be removed from the fluid domain)')                           

    my_parser.add_argument('-lpvs', '--length',
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

    my_parser.add_argument('-dt', '--time_step',
                           type=float,
                           default=1e-3,
                           help='time step')

    my_parser.add_argument('-mu', '--viscosity',
                           type=float,
                           default=7e-3,
                           help='Dynamic viscosity of the fluid')

    my_parser.add_argument('-rho', '--density',
                           type=float,
                           default=1,
                           help='Density of the fluid')

    my_parser.add_argument('-r', '--resistance',
                           type=float,
                           default=0,
                           help='Resistance at the inner side of the brain')


    my_parser.add_argument('-nr', '--N_radial',
                           type=int,
                           default=8,
                           help='number of cells in the radial direction')

    my_parser.add_argument('-nl', '--N_axial',
                           type=int,
                           default=0,
                           help='number of cells in the axial direction')

    my_parser.add_argument('-d', '--diffusion_coef',
                           type=float,
                           default=2e-8,
                           help='Diffusion coefficient of the tracer')

    my_parser.add_argument('-s', '--sigma',
                           type=float,
                           default=1e-4,
                           help='STD gaussian init for concentration')

    my_parser.add_argument('-xi', '--initial_pos',
                           type=float,
                           default=0,
                           help='Initial center of the gaussian for concentration')

    my_parser.add_argument('-c0init',
                           type=str,
                           default='null',  # gaussian, uniform
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


    my_parser.add_argument('-refineleft',
                           type=bool,
                           default=False,
                           help='Refine the mesh on the left side')

    my_parser.add_argument('-productionrate',
                           type=float,
                           default=0,
                           help='Rate of the internal production of solute')

    my_parser.add_argument('-issas',
                           type=bool,
                           default=False,
                           help='Add a SAS compartment on the left')

    my_parser.add_argument('-Lsas', '--length_sas',
                           type=float,
                           default=40e-4,
                           help='length of the SAS in the axial direction')

    my_parser.add_argument('-Rsas', '--radius_sas',
                           type=float,
                           default=50e-4,
                           help='radius of the SAS')

    my_parser.add_argument('-arigid',
                           type=float,
                           default=0,
                           help='amplitude of the rigid motion (cm)')

    my_parser.add_argument('-frigid',
                           type=float,
                           default=10,
                           help='frequency of the rigid motion (cm)')

    args = my_parser.parse_args()

    # Execute the PVS simulation

    PVS_simulation(args)