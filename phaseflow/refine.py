import fenics
import numpy


solution_at_point = numpy.array([1.e32, 1.e32, 1.e32, 1.e32, 1.e32], dtype=numpy.float_) # Oversized for up to 3D

def mark_pci_cells(regularization, mesh, w):

    hot = (regularization['theta_s'] + 2*regularization['R_s'] - fenics.dolfin.DOLFIN_EPS)
            
    cold = (regularization['theta_s'] - 2*regularization['R_s'] + fenics.dolfin.DOLFIN_EPS)

    contains_pci = fenics.CellFunction("bool", mesh)

    contains_pci.set_all(False)

    for cell in fenics.cells(mesh):
        
        hot_vertex_count = 0
        
        cold_vertex_count = 0
        
        for vertex in fenics.vertices(cell):
        
            w.eval_cell(solution_at_point, numpy.array([vertex.x(0), vertex.x(1), vertex.x(2)]), cell) # Works for 1/2/3D
            
            theta = solution_at_point[mesh.type().dim() + 1]
            
            if theta > hot:
            
                hot_vertex_count += 1
                
            if theta < cold:
            
                cold_vertex_count += 1

        if (0 < hot_vertex_count < 1 + mesh.type().dim()) | (0 < cold_vertex_count < 1 + mesh.type().dim()):
        
            contains_pci[cell] = True
                
    return contains_pci
    

''' Refine mesh near a specified boundary.
This needs to be redesigned and rewritten to be general, 
and is right now just a hack for our special cases (with unit square domain).
In deal.II I did this properly with boundary ID's'''
def refine_mesh_near_boundary(mesh, boundary, boundary_refinement_cycles):

    if mesh.type().dim() is 1: # This bandaid is for the 1D Stefan Problem.
    
        mesh = refine_mesh_near_boundary_1D(mesh, boundary_expression, boundary_refinement_cycles)
        
        return mesh
        
    else:
    
        for i in range(boundary_refinement_cycles):

            edge_markers = fenics.EdgeFunction("bool", mesh)
            
            boundary.mark(edge_markers, True)

            fenics.adapt(mesh, edge_markers)
            
            mesh = mesh.child() 
        
        return mesh
        
        
''' Refine mesh near hot boundary in 1D
The usual approach of using SubDomain and EdgeFunction isn't appearing to work
in 1D, so I'm going to just loop through the cells of the mesh and set markers manually.
'''
def refine_mesh_near_boundary_1D(mesh, boundary_refinement_cycles):

    for i in range(boundary_refinement_cycles):
    
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

    return mesh