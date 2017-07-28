import fenics
import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import phaseflow


def stefan_problem_2d_asymmetric(Ste = 1.,
        theta_h = 1.,
        theta_c = -1.,
        a_s = 2.,
        R_s = 0.005,
        dt = 0.001,
        final_time = 0.1,
        newton_relative_tolerance = 1.e-3,
        initial_uniform_cell_count = 1,
        hot_boundary_refinement_cycles = 10,
        max_pci_refinement_cycles = 10):

    mesh = fenics.UnitSquareMesh(initial_uniform_cell_count, initial_uniform_cell_count, "crossed")
    
    class Boundary(fenics.SubDomain):
        
        def inside(self, x, on_boundary):
        
            return on_boundary and (fenics.near(x[0], 0.) or fenics.near(x[1], 0.))
        
    mesh = phaseflow.refine.refine_mesh_near_boundary(
        mesh, Boundary(), hot_boundary_refinement_cycles)

    # @todo: Boundary conditions on velocity and pressure maybe unnecessary.
        
    w = phaseflow.run(
        output_dir = 'output/test_stefan_problem_2d_asymmetric_Ste'+str(Ste).replace('.', 'p')+'/',
        Pr = 1.,
        Ste = Ste,
        g = [0., 0.],
        mesh = mesh,
        max_pci_refinement_cycles = max_pci_refinement_cycles,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            "("+str(theta_h)+" - "+str(theta_c)+")*(near(x[0],  0.) | near(x[1], 0.)) + "+str(theta_c)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0., 0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_h, 'degree': 2, 'location_expression': "near(x[0],  0.) | near(x[1], 0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_c, 'degree': 2, 'location_expression': "near(x[0],  1.) | near(x[1], 1.)", 'method': "topological"}],
        regularization = {'a_s': 2., 'theta_s': 0.01, 'R_s': R_s},
        newton_relative_tolerance = newton_relative_tolerance,
        final_time = final_time,
        time_step_bounds = dt,
        linearize = False)
        
    return w


def test_stefan_problem_2d_asymmetric():

    for p in [{'Ste': 0.1, 'R_s': 0.05, 'dt': 0.001, 'final_time': 0.1,
               'initial_uniform_cell_count': 10, 
               'hot_boundary_refinement_cycles': 5,
               'newton_relative_tolerance': 1.e-3}]:
        
        w = stefan_problem_2d_asymmetric(
            Ste=p['Ste'],
            R_s=p['R_s'],
            dt=p['dt'],
            final_time = p['final_time'],
            initial_uniform_cell_count=p['initial_uniform_cell_count'],
            hot_boundary_refinement_cycles=p['hot_boundary_refinement_cycles'],
            newton_relative_tolerance=p['newton_relative_tolerance'])     

        
if __name__=='__main__':
    
    test_stefan_problem_2d_asymmetric()
