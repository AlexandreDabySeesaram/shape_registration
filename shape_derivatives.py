import dolfin
import ufl
import numpy as np

#%% Define projection of the image onto the domain mesh

def I(mesh, image_expression):
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    
    i_proj = dolfin.project(image_expression, V)
    
    return i_proj

def I_3D(mesh, image_expression):
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    
    i_proj = dolfin.interpolate(image_expression, V)
    
    return i_proj

def grad_I_3D(mesh, image_expression):
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    
    i_proj = dolfin.interpolate(image_expression, V)
    
    return dolfin.grad(i_proj)

def grad_I(mesh, image_expression):
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    
    i_proj = dolfin.project(image_expression, V)
    
    return dolfin.grad(i_proj)

def int_I(mesh, image_expression):
    i_proj = I(mesh, image_expression)
    
    return dolfin.assemble(i_proj * dolfin.dx)

def int_I_3D(mesh, image_expression):
    i_proj = I_3D(mesh, image_expression)
    
    return dolfin.assemble(i_proj * dolfin.dx)



def shape_derivative_volume(mesh, u_current, I, grad_I, alpha=1, gamma=10):
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    
    u, v = dolfin.TestFunction(V), dolfin.TrialFunction(V)    
    shape_derivative = dolfin.div(v) * I * dolfin.dx + dolfin.inner(grad_I, v) * dolfin.dx
    # Regularization term (choice of inner_product)
    inner_product = dolfin.inner(dolfin.grad(u) + dolfin.grad(u).T, dolfin.grad(v)) * dolfin.dx + alpha * dolfin.inner(u, v) * dolfin.dx

    # Solve the system to find shapeGrad
    shape_gradient = dolfin.Function(V)
    dolfin.solve(inner_product == shape_derivative, shape_gradient)
    return shape_gradient.vector()[:]




def check_flipped_triangles(mesh):

    det_J = np.zeros(mesh.num_cells())
    for i, cell in enumerate(dolfin.cells(mesh)):
        # Compute the Jacobian determinant of the cell
        det_J_i = cell.volume()
        det_J[i] = det_J_i
        if det_J_i < 0:
            print("negative jac")
            return 1
    negative_count = np.sum(det_J < 0)

    return negative_count>0


def update_GD(mesh, image, u, descentDir, step=1, minStep=1e-6):
    
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    
    delta_u = dolfin.Function(V)
    new_u = dolfin.Function(V)
    delta_u.vector()[:] = 0  

    # Compute the functional int_I at the current mesh state
    old_int_I = int_I(mesh, image)
    new_int_I = old_int_I + 1  # Set to ensure the loop runs at least once

    # Start Armijo backtracking with an initial step size
    step = step * 2
    while new_int_I > old_int_I:
        step = step / 2  

        delta_u.vector()[:] = -delta_u.vector()[:]
        dolfin.ALE.move(mesh, (delta_u))                           # revert change

        # Apply the new deformation direction with the step size
        delta_u.vector()[:] = step * descentDir
        # Deform the mesh (apply the deformation to the mesh)
        dolfin.ALE.move(mesh, delta_u)
        
        new_u.vector()[:] = u.vector()[:] + delta_u.vector()[:]

        # Check for flipped triangles (you would need to implement your own logic here)
        nFlipped = check_flipped_triangles(mesh)
        if nFlipped > 0:
            print(f"{nFlipped} flipped triangles found, decreasing step size...")
            continue  # Backtrack if flipped triangles are found
        
        # Evaluate the functional at the new deformation
        new_int_I = int_I(mesh, image)
        
        if step <= minStep:
            print(f"Backtracking failed, step <= {minStep}")
            break
    
    # Return the updated deformation, the new value of the functional, and the step size
    return new_u, new_int_I, step


def update_GD_3D(mesh, image, u, descentDir, step=1, minStep=1e-6):
    
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    
    delta_u = dolfin.Function(V)
    new_u = dolfin.Function(V)
    delta_u.vector()[:] = 0  

    # Compute the functional int_I at the current mesh state
    old_int_I = int_I_3D(mesh, image)
    new_int_I = old_int_I + 1  # Set to ensure the loop runs at least once

    # Start Armijo backtracking with an initial step size
    step = step * 2
    while new_int_I > old_int_I:
        step = step / 2  

        delta_u.vector()[:] = -delta_u.vector()[:]
        dolfin.ALE.move(mesh, (delta_u))                           # revert change

        # Apply the new deformation direction with the step size
        delta_u.vector()[:] = step * descentDir
        # Deform the mesh (apply the deformation to the mesh)
        dolfin.ALE.move(mesh, delta_u)
        
        new_u.vector()[:] = u.vector()[:] + delta_u.vector()[:]

        # Check for flipped triangles (you would need to implement your own logic here)
        nFlipped = check_flipped_triangles(mesh)
        if nFlipped > 0:
            print(f"{nFlipped} flipped triangles found, decreasing step size...")
            continue  # Backtrack if flipped triangles are found
        
        # Evaluate the functional at the new deformation
        new_int_I = int_I_3D(mesh, image)
        
        if step <= minStep:
            print(f"Backtracking failed, step <= {minStep}")
            break
    
    # Return the updated deformation, the new value of the functional, and the step size
    return new_u, new_int_I, step