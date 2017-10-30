""" This is a MWE to demonstrate a bug
where adaptivenonlinearvariationalsolver produces NaN's
"""
import fenics


def write_solution(solution_file, w, time):
    """Write the solution to disk."""
    print("Writing solution to HDF5+XDMF")
    
    velocity, pressure, temperature = w.leaf_node().split()
    
    velocity.rename("u", "velocity")
    
    pressure.rename("p", "pressure")
    
    temperature.rename("T", "temperature")
    
    for var in [velocity, pressure, temperature]:
    
        solution_file.write(var, time)

        
class Model():

    def __init__(self, mesh, pressure_degree, temperature_degree,
            initial_values, boundary_conditions, time, time_step_size,
            liquid_viscosity, rayleigh_number, prandtl_number, stefan_number,
            gravity, temperature_of_fusion, smoothing_radius):
    
        self.mesh = mesh
        
        self.pressure_degree = pressure_degree
        
        self.temperature_degree = temperature_degree
        
        self.velocity_degree = self.pressure_degree + 1
        
        self.W_ele = self.make_mixed_fe(mesh)
        
        self.W = fenics.FunctionSpace(self.mesh, self.W_ele)
        
        self.w_k = fenics.Function(self.W)
        
        self.w_n = fenics.Expression(initial_values, element=self.W_ele)
        
        self.bc_dicts = boundary_conditions
        
        self.bcs = []
        
        self.update_bcs()
        
        self.time = time
        
        self.time_step_size = time_step_size
        
        self.liquid_viscosity = liquid_viscosity
        
        self.solid_viscosity = 1.e4
        
        self.reynolds_number = 1.
        
        self.rayleigh_number = rayleigh_number
        
        self.prandtl_number = prandtl_number
        
        self.stefan_number = stefan_number
        
        self.gravity = gravity

        self.heat_capacity = 1.

        self.thermal_conductivity = 1.

        self.penalty_factor = 1.e-7

        self.temperature_of_fusion = temperature_of_fusion
        
        self.smoothing_radius = smoothing_radius
        
        
    def make_mixed_fe(self, mesh):
        """ Define mixed finite element.
        MixedFunctionSpace used to be available but is now deprecated. 
        To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
        """
        velocity_element = fenics.VectorElement('P', mesh.ufl_cell(),
            self.velocity_degree)

        pressure_element = fenics.FiniteElement('P', mesh.ufl_cell(),
            self.pressure_degree)

        temperature_element = fenics.FiniteElement('P', mesh.ufl_cell(),
            self.temperature_degree)

        solution_element = fenics.MixedElement([velocity_element, pressure_element, 
            temperature_element])
        
        return solution_element
        
        
    def update_bcs(self):
        """ Parse boundary condition dictionaries and apply them in the space W."""
        self.bcs = []
    
        for item in self.bc_dicts:

            self.bcs.append(fenics.DirichletBC(self.W.sub(item['subspace']),
                item['value'], item['location'], method=item['method']))
                
        
    def solve(self, adaptive, adaptive_tolerance=None, newton_relative_tolerance=1.e-8):
        """ Define and solve the variational problem. """
        
        mu_l = fenics.Constant(self.liquid_viscosity)
        
        mu_s = fenics.Constant(self.solid_viscosity)
        
        Ra = fenics.Constant(self.rayleigh_number)
        
        Pr = fenics.Constant(self.prandtl_number)
        
        Re = fenics.Constant(self.reynolds_number)
        
        g = fenics.Constant(self.gravity)
        
        dt = fenics.Constant(self.time_step_size)
        
        C = fenics.Constant(self.heat_capacity)

        K = fenics.Constant(self.thermal_conductivity)

        Ste = fenics.Constant(self.stefan_number)

        gamma = fenics.Constant(self.penalty_factor)
        
        T_f = fenics.Constant(self.temperature_of_fusion)
        
        r = fenics.Constant(self.smoothing_radius)
        
        inner, dot, grad, div = fenics.inner, fenics.dot, fenics.grad, fenics.div
        
        sym = fenics.sym
        
        D = lambda u : sym(grad(u))
        
        a = lambda mu, u, v : 2.*mu*inner(D(u), D(v))
        
        b = lambda u, q : -div(u)*q
        
        c = lambda w, z, v : dot(dot(grad(z), w), v)
        
        f_B = lambda T : g*T*Ra/(Pr*Re**2)
        
        L = C/Ste
        
        P = lambda T: 0.5*(1. - fenics.tanh(2.*(T_f - T)/r))
        
        mu = lambda (T) : mu_s + (mu_l - mu_s)*P(T)
        
        u_n, p_n, T_n = fenics.split(self.w_n)

        w_w = fenics.TrialFunction(self.W)
        
        u_w, p_w, T_w = fenics.split(w_w)
        
        v, q, phi = fenics.TestFunctions(self.W)
        
        u_k, p_k, T_k = fenics.split(self.w_k)

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
        
        
        # Set the Gateaux derivative.
        df_B = lambda T : g*Ra/(Pr*Re**2)

        sech = lambda theta: 1./fenics.cosh(theta)
        
        dP = lambda T: sech(2.*(T_f - T)/r)**2/r

        dmu = lambda T : (mu_l - mu_s)*dP(T)
        
        JF = (
            b(u_w, q) - gamma*p_w*q 
            + dot(u_w, v)/dt
            + c(u_k, u_w, v) + c(u_w, u_k, v) + b(v, p_w)
            + a(T_w*dmu(T_k), u_k, v) + a(mu(T_k), u_w, v) 
            + dot(T_w*df_B(T_k), v)
            + C/dt*T_w*phi
            - dot(C*T_k*u_w, grad(phi))
            - dot(C*T_w*u_k, grad(phi))
            + K/Pr*dot(grad(T_w), grad(phi))
            + 1./dt*L*T_w*dP(T_k)*phi
            )*fenics.dx

            
        # Make the problem.
        problem = fenics.NonlinearVariationalProblem(F, self.w_k, self.bcs, JF)
        
        
        # Solve the problem.
        if adaptive:

            M = P(T_k)*fenics.dx  # Adaptive goal functional
    
            solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)
            
            solver.parameters['nonlinear_variational_solver']['newton_solver']['relative_tolerance'] = \
                newton_relative_tolerance
            
            solver.solve(adaptive_tolerance)
            
        else:
        
            solver = fenics.NonlinearVariationalSolver(problem)
            
            solver.parameters['newton_solver']['relative_tolerance'] = \
                newton_relative_tolerance
            
            solver.solve()
            
        self.time += self.time_step_size
        
        
        # Update initial values for the next time step.
        self.w_n = self.w_k.leaf_node().copy()
    
        """If the mesh was refined during adaptive solving, then update the mesh, solution space,
        solution function, and boundary conditions.
        """
        if adaptive and (self.w_k.depth > 1):
        
            self.mesh = self.mesh.leaf_node()
            
            self.W = fenics.FunctionSpace(self.mesh, self.W_ele)
            
            self.w_k = fenics.Function(self.W)
            
            self.update_bcs()
            

def test_adaptive_natural_convection_in_differentially_heated_cavity():
    
    m = 10
    
    mesh = fenics.UnitSquareMesh(m, m, 'crossed')
    
    T_hot, T_cold = 0.5, -0.5

    start_time, end_time = 0., 10.
    
    time_step_size = 1.e-3
    
    model = Model(mesh = mesh,
        pressure_degree = 1,
        temperature_degree = 1,
        initial_values = ("0.", "0.", "0.",
            str(T_hot)+"*near(x[0],  0.) "+str(T_cold)+"*near(x[0],  1.)"),
        boundary_conditions = [
            {'subspace': 0, 'value': ("0.", "0."),
            'location': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)",
            'method': 'topological'},
            {'subspace': 2, 'value': str(T_hot),
            'location': "near(x[0],  0.)",
            'method': 'topological'},
            {'subspace': 2, 'value': str(T_cold),
            'location': "near(x[0],  1.)",
            'method': 'topological'}],
        time=start_time, time_step_size=time_step_size, liquid_viscosity=1.,
        rayleigh_number=1.e6, prandtl_number=0.71, stefan_number = 1.,
        gravity=(0., -1.),
        temperature_of_fusion=-1., smoothing_radius=0.05)
    
    progress = fenics.Progress('Time-stepping')

    fenics.set_log_level(fenics.PROGRESS)
        
    with fenics.XDMFFile('output/natural_convection/solution.xdmf') as solution_file:
    
    
        # Write the initial values
        write_solution(solution_file, model.w_n, model.time)
            
            
        # Solve a sequence of time steps.
        while model.time < (end_time - fenics.DOLFIN_EPS):
        
            model.solve(adaptive=True, adaptive_tolerance=1.e-8)

            write_solution(solution_file, model.w_k, model.time)
            
            progress.update(model.time / end_time)
            
            
            # Double the time step size, which had to be small in the beginning.
            model.time_step_size *= 2
    
    
def lid_driven_cavity(adaptive):

    lid = 'near(x[1],  1.)'

    fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'

    bottom_left_corner = 'near(x[0], 0.) && near(x[1], 0.)'
    
    m = 3
    
    mesh = fenics.UnitSquareMesh(m, m, 'crossed')
  
    model = Model(mesh=mesh,
        pressure_degree = 1,
        temperature_degree = 1,
        initial_values = (lid, "0.", "0.", "0."), 
        boundary_conditions = [
            {'subspace': 0, 'value': ("1.", "0."), 'location': lid,
                'method': 'topological'},
            {'subspace': 0, 'value': ("0.", "0."), 'location': fixed_walls,
                'method': 'topological'},
            {'subspace': 1, 'value': "0.", 'location': bottom_left_corner,
                'method': 'pointwise'}],
        time=0., time_step_size=1.e12, liquid_viscosity=0.01,
        rayleigh_number=1., prandtl_number=1., stefan_number=1.,
        gravity=(0., 0.),
        temperature_of_fusion=-1., smoothing_radius = 0.05)

    model.solve(adaptive=adaptive, adaptive_tolerance=1.e-8)
        
    with fenics.XDMFFile('output/lid_driven_cavity/solution.xdmf') as solution_file:
    
        write_solution(solution_file=solution_file, w=model.w_k, time=model.time)
    
    
def test_lid_driven_cavity():

    lid_driven_cavity(adaptive=False)
    
    
def test_adaptive_lid_driven_cavity():

    lid_driven_cavity(adaptive=True)
    
    
def test_adaptive_melting_pcm():
    
    
    # Make the mesh
    m = 20
    
    mesh = fenics.UnitSquareMesh(m, m, 'crossed')
    
    initial_hot_wall_refinement_cycles = 2
    
    class HotWall(fenics.SubDomain):
        
        def inside(self, x, on_boundary):
        
            return on_boundary and fenics.near(x[0], 0.)

            
    hot_wall = HotWall()
    
    for i in range(initial_hot_wall_refinement_cycles):
        
        edge_markers = fenics.EdgeFunction("bool", mesh)
        
        hot_wall.mark(edge_markers, True)

        fenics.adapt(mesh, edge_markers)
        
        mesh = mesh.child()
    
    
    # Make the model.
    T_hot, T_cold = 1., -0.1
    
    model = Model(mesh=mesh,
        pressure_degree = 1,
        temperature_degree = 1,
        initial_values=("0.", "0.", "0.",
            "("+str(T_hot)+" - "+str(T_cold)+")*(x[0] < 0.001) + "+str(T_cold)),
        boundary_conditions = [
            {'subspace': 0, 'value': ("0.", "0."),
            'location': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)",
            'method': 'topological'},
            {'subspace': 2, 'value': str(T_hot),
            'location': "near(x[0],  0.)",
            'method': 'topological'},
            {'subspace': 2, 'value': str(T_cold),
            'location': "near(x[0],  1.)",
            'method': 'topological'}],
        time=0., time_step_size=1.e-3, liquid_viscosity=1.,
        rayleigh_number=1., prandtl_number=1., stefan_number=1.,
        gravity=(0., -1.),
        temperature_of_fusion=0.1, smoothing_radius=0.05)
    
    
    # Solve a sequence of time steps.
    progress = fenics.Progress('Time-stepping')

    fenics.set_log_level(fenics.PROGRESS)
        
    with fenics.XDMFFile('output/melting_pcm/solution.xdmf') as solution_file:
    
        write_solution(solution_file, fenics.interpolate(model.w_n, model.W), model.time)
            
        end_time = 0.01
        
        while model.time < (end_time - fenics.DOLFIN_EPS):
        
            model.solve(adaptive=True, adaptive_tolerance=1.e-4)
            
            print("Reached time t = ", str(model.time))

            write_solution(solution_file, model.w_n, model.time)
            
            progress.update(model.time / end_time)
            
            
def test_1d_stefan_problem():

    def write_solution(solution_file, w, W, time):

        u, p, T = w.leaf_node().split()
        
        T.rename("T", "temperature")
        
        coordinates = W.tabulate_dof_coordinates()
        
        for x in coordinates:
        
            solution_file.write(str(time)+", "+str(x)+", "+str(T(x))+"\n")

    
    mesh = fenics.UnitIntervalMesh(10)
    
    T_hot, T_cold = 1., -1
    
    
    # Refine the initial mesh near the hot wall
    initial_hot_wall_refinement_cycles = 10
    
    ''' Refine mesh near hot boundary
    The usual approach of using SubDomain and EdgeFunction isn't appearing to work
    in 1D, so I'm going to just loop through the cells of the mesh and set markers manually.
    '''
    for i in range(initial_hot_wall_refinement_cycles):
    
        cell_markers = fenics.CellFunction("bool", mesh)
        
        cell_markers.set_all(False)
        
        for cell in fenics.cells(mesh):
        
            found_hot_boundary = False
        
            for vertex in fenics.vertices(cell):
            
                if fenics.near(vertex.x(0), 0., fenics.dolfin.DOLFIN_EPS):
                
                    found_hot_boundary = True
                    
            if found_hot_boundary:
            
                cell_markers[cell] = True

                break # There should only be one such point.
                
        mesh = fenics.refine(mesh, cell_markers)
    
    
    # Make the model.
    model = Model(mesh=mesh,
        pressure_degree = 1,
        temperature_degree = 1,
        initial_values=("0.", "0.",
            "("+str(T_hot)+" - "+str(T_cold)+")*near(x[0], 0.) + "+str(T_cold)),
        boundary_conditions = [
            {'subspace': 0, 'value': [0.],
            'location': "near(x[0],  0.) | near(x[0],  1.)", 'method': 'topological'},
            {'subspace': 2, 'value': T_hot,
            'location': "near(x[0],  0.)", 'method': 'topological'},
            {'subspace': 2, 'value': T_cold,
            'location': "near(x[0],  1.)", 'method': 'topological'}],
        time=0., time_step_size=1.e-4, liquid_viscosity=1.,
        rayleigh_number=1., prandtl_number=1., stefan_number=0.1,
        gravity=[0.],
        temperature_of_fusion=0.01, smoothing_radius=0.05)
    
    progress = fenics.Progress('Time-stepping')

    fenics.set_log_level(fenics.PROGRESS)
    
    with open('output/stefan_problem/solution.csv', 'w+') as solution_file:
    
        solution_file.write("t, x, T\n")
    
        # Write the initial values
        write_solution(solution_file, fenics.interpolate(model.w_n, model.W),
            model.W, model.time)
        
        
        # Solve a sequence of time steps.
        end_time = 0.01
        
        while model.time < (end_time - fenics.DOLFIN_EPS):
        
            model.solve(adaptive=True, adaptive_tolerance=1.e-4, newton_relative_tolerance=1.e-4)

            write_solution(solution_file, model.w_k, model.W, model.time)
            
            progress.update(model.time / end_time)
            
    
if __name__=='__main__':
    
    #test_1d_stefan_problem()
    
    #test_adaptive_natural_convection_in_differentially_heated_cavity()
    
    #test_lid_driven_cavity()
    
    #test_adaptive_lid_driven_cavity()
    
    test_adaptive_melting_pcm()
    