import fenics
import phaseflow
import scipy.optimize as opt

   
def extract_pci_position(w):

    def theta(x):
    
        wval = w(fenics.Point(x))
        
        return wval[2]
    
    pci_pos = opt.newton(theta, 0.1)
    
    return pci_pos
    
    
def stefan_problem_solidify(Ste = 0.125,
    T_h = 0.01,
    T_c = -1.,
    T_f = 0.,
    r = 0.01,
    dt = 0.01,
    end_time = 1.,
    initial_uniform_cell_count = 100):

    
    mesh = fenics.UnitIntervalMesh(initial_uniform_cell_count)

    w, mesh = phaseflow.run(
        output_dir = 'output/convergence_stefan_problem_solidify_vary_r/r'+str(r)+'dt'+str(dt)+
            '/dx'+str(1./float(initial_uniform_cell_count))+'/',
        prandtl_number = 1.,
        stefan_number = Ste,
        gravity = [0.],
        mesh = mesh,
        initial_values_expression = (
            "0.",
            "0.",
            "("+str(T_c)+" - "+str(T_h)+")*near(x[0],  0.) + "+str(T_h)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': T_c, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': T_h, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        temperature_of_fusion = T_f,
        regularization_smoothing_factor = r,
        end_time = end_time,
        time_step_size = dt)
        
    return w
  

def convergence_stefan_problem_1d():

    phaseflow.helpers.mkdir_p('output/convergence_stefan_problem_solidify_vary_r/')
    
    with open('output/convergence_stefan_problem_solidify_vary_r/convergence.txt',
            'a+') as file:
    
        file.write("regularization_smoothing_factor,dt,dx,pci_pos\n")
        
        nt = 200

        for r in [0.005, 0.01, 0.02, 0.04, 0.08]:
        
            for nx in [400, 800, 1600]:
            
                dt = 1./float(nt)
                
                dx = 1./float(nx)
            
                w = stefan_problem_solidify(r=r, dt = 1./float(nt), initial_uniform_cell_count = nx)
            
                pci_pos = extract_pci_position(w)
                
                file.write(str(r) + "," + str(dt) + "," + str(dx) + "," + str(pci_pos) + "\n")

            
if __name__=='__main__':
    
    convergence_stefan_problem_1d()
