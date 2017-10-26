"""This module contains the core functionalty of Phaseflow."""
import fenics
import h5py
import helpers
import globals
import default
import form
import solver
import bounded_value
import time
import refine
import output


def function_spaces(mesh=default.mesh, pressure_degree=default.pressure_degree, temperature_degree=default.temperature_degree):
    """ Define function spaces for the variational form."""
    
    velocity_degree = pressure_degree + 1

    ''' MixedFunctionSpace used to be available but is now deprecated. 
    The way that fenics separates function spaces and elements is confusing.
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
    '''
    velocity_element = fenics.VectorElement('P', mesh.ufl_cell(), velocity_degree)

    pressure_element = fenics.FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    temperature_element = fenics.FiniteElement('P', mesh.ufl_cell(), temperature_degree)

    solution_element = fenics.MixedElement([velocity_element, pressure_element, temperature_element])

    solution_function_space = fenics.FunctionSpace(mesh, solution_element)  
    
    return solution_function_space, solution_element


def run(
    output_dir = 'output/wang2010_natural_convection_air',
    Ra = default.parameters['Ra'],
    Pr = default.parameters['Pr'],
    Ste = default.parameters['Ste'],
    C = default.parameters['C'],
    K = default.parameters['K'],
    mu_l = default.parameters['mu_l'],
    mu_s = default.parameters['mu_s'],
    g = default.parameters['g'],
    m_B = default.m_B,
    ddT_m_B = default.ddT_m_B,
    regularization = default.regularization,
    mesh=default.mesh,
    initial_values_expression = ("0.", "0.", "0.", "0.5*near(x[0],  0.) -0.5*near(x[0],  1.)"),
    boundary_conditions = [
        {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3,
        'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)", 'method': 'topological'},
        {'subspace': 2, 'value_expression': "0.5", 'degree': 2, 
        'location_expression': "near(x[0],  0.)", 'method': 'topological'},
        {'subspace': 2, 'value_expression': "-0.5", 'degree': 2, 
        'location_expression': "near(x[0],  1.)", 'method': 'topological'}],
    start_time = 0.,
    end_time = 10.,
    time_step_size = 1.e-3,
    max_time_steps = 1000,
    gamma = 1.e-7,
    adaptive_solver_tolerance = 1.e-4,
    nlp_relative_tolerance = 1.e-4,
    nlp_max_iterations = 12,
    pressure_degree = default.pressure_degree,
    temperature_degree = default.temperature_degree,
    stop_when_steady = True,
    steady_relative_tolerance = 1.e-4,
    restart = False,
    restart_filepath = '',
    debug = False):
    """Run Phaseflow.
    
    Rather than using an input file, Phaseflow is configured entirely through
    the arguments in this run() function.
    
    See the tests and examples for demonstrations of how to use this.
    """
    
    '''@todo Describe the arguments in the docstring.
    Phaseflow has been in rapid development and these have been changing.
    Now that things are stabilizing somewhat, it's about time to document
    these arguments properly.
    '''
    
    # Display inputs
    helpers.print_once("Running Phaseflow with the following arguments:")
    
    helpers.print_once(helpers.arguments())
    
    helpers.mkdir_p(output_dir)
    
    if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
        arguments_file = open(output_dir + '/arguments.txt', 'w')
        
        arguments_file.write(str(helpers.arguments()))

        arguments_file.close()
    
    
    #
    dimensionality = mesh.type().dim()
    
    helpers.print_once("Running "+str(dimensionality)+"D problem")
    
    
    # Initialize time
    if restart:
    
        with h5py.File(restart_filepath, 'r') as h5:
            
            current_time = h5['t'].value
            
            assert(abs(current_time - start_time) < time.TIME_EPS)
    
    else:
    
        current_time = start_time
    
    
    # Define function spaces and solution function 
    W, W_ele = function_spaces(mesh, pressure_degree, temperature_degree)
    
    
    # Set the initial values
    if restart:
            
        mesh = fenics.Mesh()
        
        with fenics.HDF5File(mesh.mpi_comm(), restart_filepath, 'r') as h5:
        
            h5.read(mesh, 'mesh', True)
        
        W, W_ele = function_spaces(mesh, pressure_degree, temperature_degree)
    
        w_n = fenics.Function(W)
    
        with fenics.HDF5File(mesh.mpi_comm(), restart_filepath, 'r') as h5:
        
            h5.read(w_n, 'w')
    
    else:

        w_n = fenics.Expression(initial_values_expression, element=W_ele)

        
    # Organize boundary conditions
    bcs = []
    
    for item in boundary_conditions:
    
        bcs.append(fenics.DirichletBC(W.sub(item['subspace']), item['value_expression'],
            item['location_expression'], method=item['method']))
    
    
    # Set the variational form
    """Set local names for math operators to improve readability."""
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym


    """Next we write the linear, bilinear, and trilinear forms.

    These follow the common notation for applying the finite element method
    to the incompressible Navier-Stokes equations, e.g. from danaila2014newton
    and huerta2003fefluids.
    """

    """The bilinear form for the stress-strain matrix in Stokes flow."""
    def a(mu, u, v):

        def D(u):
        
            return sym(grad(u))
        
        return 2.*mu*inner(D(u), D(v))


    """The linear form for the divergence in incompressible flow."""
    def b(u, q):
        
        return -div(u)*q
        

    """The trilinear form for convection of the velocity field."""
    def c(w, z, v):
       
        return dot(dot(grad(z), w), v)
    
    
    """Time step size."""
    dt = fenics.Constant(time_step_size)
    
    """Rayleigh Number"""
    Ra = fenics.Constant(Ra), 
    
    """Prandtl Number"""
    Pr = fenics.Constant(Pr)
    
    """Stefan Number"""
    Ste = fenics.Constant(Ste)
    
    """Heat capacity"""
    C = fenics.Constant(C)
    
    """Thermal diffusivity"""
    K = fenics.Constant(K)
    
    """Gravity"""
    g = fenics.Constant(g)
    
    """Parameter for penalty formulation
    of incompressible Navier-Stokes"""
    gamma = fenics.Constant(gamma)
    
    """Liquid viscosity"""
    mu_l = fenics.Constant(mu_l)
    
    """Solid viscosity"""
    mu_s = fenics.Constant(mu_s)
    
    """Buoyancy force, $f = ma$"""
    f_B = lambda T : m_B(T)*g
    
    """Parameter shifting the tanh regularization"""
    T_f = fenics.Constant(regularization['T_f'])
    
    """Parameter scaling the tanh regularization"""
    r = fenics.Constant(regularization['r'])
    
    """Latent heat"""
    L = C/Ste
    
    """Regularize heaviside function with a 
    hyperoblic tangent function."""
    P = lambda T: 0.5*(1. - fenics.tanh(2.*(T_f - T)/r))
    
    """Variable viscosity"""
    mu = lambda (T) : mu_s + (mu_l - mu_s)*P(T)
    
    """Set the nonlinear variational form."""
    u_n, p_n, T_n = fenics.split(w_n)

    w_w = fenics.TrialFunction(W)
    
    u_w, p_w, T_w = fenics.split(w_w)
    
    v, q, phi = fenics.TestFunctions(W)
    
    w_k = fenics.Function(W)
    
    u_k, p_k, T_k = fenics.split(w_k)

    F = (
        b(u_k, q) - gamma*p_k*q
        + dot(u_k - u_n, v)/dt
        + c(u_k, u_k, v) + b(v, p_k) + a(mu(T_k), u_k, v)
        + dot(f_B(T_k), v)
        + C/dt*(T_k - T_n)*phi
        - dot(C*T_k*u_k, grad(phi)) 
        + K/Pr*dot(grad(T_k), grad(phi))
        + 1./dt*L*(P(T_k) - P(T_n))*phi
        )*fenics.dx

    """Set the Jacobian (formally the Gateaux derivative)."""
    ddT_f_B = lambda T : ddT_m_B(T)*g
    
    sech = lambda theta: 1./fenics.cosh(theta)
    
    dP = lambda T: sech(2.*(T_f - T)/r)**2/r

    dmu = lambda T : (mu_l - mu_s)*dP(T)

    JF = (
        b(u_w, q) - gamma*p_w*q 
        + dot(u_w, v)/dt
        + c(u_k, u_w, v) + c(u_w, u_k, v) + b(v, p_w)
        + a(T_w*dmu(T_k), u_k, v) + a(mu(T_k), u_w, v) 
        + dot(T_w*ddT_f_B(T_k), v)
        + C/dt*T_w*phi
        - dot(C*T_k*u_w, grad(phi))
        - dot(C*T_w*u_k, grad(phi))
        + K/Pr*dot(grad(T_w), grad(phi))
        + 1./dt*L*T_w*dP(T_k)*phi
        )*fenics.dx

    """ Set goal functional for adaptive mesh refinement"""
    M = (u_k[0] + T_k + P(T_k))*fenics.dx
    
    #
    problem = fenics.NonlinearVariationalProblem(F, w_k, bcs, JF)
    
    
    # Make the solver
    solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)
    
    solver.parameters['nonlinear_variational_solver']['newton_solver']['maximum_iterations'] = nlp_max_iterations
    
    solver.parameters['nonlinear_variational_solver']['newton_solver']['relative_tolerance'] = nlp_relative_tolerance

    solver.parameters['nonlinear_variational_solver']['newton_solver']['error_on_nonconvergence'] = True

    
    ''' @todo  explore info(f.parameters, verbose=True) 
    to avoid duplicate mesh storage when appropriate 
    per https://fenicsproject.org/qa/3051/parallel-output-of-a-time-series-in-hdf5-format '''
    
    with fenics.XDMFFile(output_dir + '/solution.xdmf') as solution_file:

        # Write the initial values
        output.write_solution(solution_file,
            fenics.interpolate(w_n, W), current_time)
    
    
        # Solve each time step
        progress = fenics.Progress('Time-stepping')

        fenics.set_log_level(fenics.PROGRESS)

        if stop_when_steady:
        
            steady = False
        
        for it in range(max_time_steps):
            
            if start_time >= end_time - time.TIME_EPS:
            
                helpers.print_once("Start time is already too close to end time. Only writing initial values.")
                
                fe_field_interpolant = fenics.interpolate(w_n.leaf_node(), W.leaf_node())
                
                return fe_field_interpolant, mesh
            
            solver.solve(adaptive_solver_tolerance)
            
            current_time += time_step_size
            
            if it == 0:
            
                w_n = fenics.interpolate(w_n, W.leaf_node())
            
            if stop_when_steady and time.steady(W, w_k, w_n, 
                    steady_relative_tolerance):
            
                steady = True
            
            output.write_solution(solution_file, w_k, current_time)
            
            
            # Write checkpoint/restart files
            restart_filepath = output_dir+'/restart_t'+str(current_time)+'.h5'
            
            with fenics.HDF5File(fenics.mpi_comm_world(), restart_filepath, 'w') as h5:
    
                h5.write(mesh.leaf_node(), 'mesh')
            
                h5.write(w_k.leaf_node(), 'w')
                
            if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
            
                with h5py.File(restart_filepath, 'r+') as h5:
                    
                    h5.create_dataset('t', data=current_time)
                        
            helpers.print_once("Reached time t = " + str(current_time))
                
            if stop_when_steady and steady:
            
                helpers.print_once("Reached steady state at time t = " + str(current_time))
                
                break

            w_n.leaf_node().vector()[:] = w_k.leaf_node().vector()  # The current solution becomes the new initial values
            
            progress.update(current_time / end_time)
            
            if current_time >= (end_time - fenics.dolfin.DOLFIN_EPS):
            
                helpers.print_once("Reached end time, t = "+str(end_time))
            
                break
                
    # Return the interpolant to sample inside of Python
    w_n.rename('w', "state")
        
    fe_field_interpolant = fenics.interpolate(w_n.leaf_node(), W)
    
    return fe_field_interpolant, mesh
    
    
if __name__=='__main__':

    run()
