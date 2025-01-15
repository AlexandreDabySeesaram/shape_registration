from dolfin import *
import ufl
import numpy as np

#%% Define projection of the image onto the domain mesh

def I(mesh, image_expression):
    V = FunctionSpace(mesh, "CG", 1)
    
    i_proj = project(image_expression, V)
    
    return i_proj

def I_3D(mesh, image_expression):
    V = FunctionSpace(mesh, "CG", 1)
    
    i_proj = interpolate(image_expression, V)
    
    return i_proj

def grad_I_3D(mesh, image_expression):
    V = FunctionSpace(mesh, "CG", 1)
    
    i_proj = interpolate(image_expression, V)
    
    return grad(i_proj)

def grad_I(mesh, image_expression):
    V = FunctionSpace(mesh, "CG", 1)
    
    i_proj = project(image_expression, V)
    
    return grad(i_proj)

def int_I(mesh, image_expression):
    i_proj = I(mesh, image_expression)
    
    return assemble(i_proj * dx)

def int_I_3D(mesh, image_expression):
    i_proj = I_3D(mesh, image_expression)
    
    return assemble(i_proj * dx)



def shape_derivative_volume(mesh, u_current, I, grad_I, alpha=1, gamma=10):
    V = VectorFunctionSpace(mesh, "CG", 1)
    
    u, v = TestFunction(V), TrialFunction(V)

    # Deform the mesh by u_current
    # mesh = Mesh(mesh_0)

    # ALE.move(mesh, u_current)        
    
    shape_derivative = div(v) * I * dx + inner(grad_I, v) * dx


    # Regularization term (choice of inner_product)
    inner_product = inner(grad(u) + grad(u).T, grad(v)) * dx + alpha * inner(u, v) * dx



    
    # Cauchy-Riemann term for preserving mesh quality
    # inner_product += gamma * (u.dx(0) - u.dx(1)) * (v.dx(0) - v.dx(1)) * dx
    # inner_product += gamma * (u.dx(1) + u.dx(0)) * (v.dx(1) + v.dx(0)) * dx

    # Solve the system to find shapeGrad
    shape_gradient = Function(V)
    solve(inner_product == shape_derivative, shape_gradient)
    return shape_gradient.vector()[:]




def check_flipped_triangles(mesh):

    det_J = np.zeros(mesh.num_cells())
    for i, cell in enumerate(cells(mesh)):
        # Compute the Jacobian determinant of the cell
        det_J_i = cell.volume()
        det_J[i] = det_J_i
        if det_J_i < 0:
            print("negative jac")
            return 1
    negative_count = np.sum(det_J < 0)
    # print(f"negative_count is {negative_count}")
    # print(f"min J is {min(det_J)}")
    # print(f"max J is {max(det_J)}")

    return negative_count>0


def update_GD(mesh, mesh_0, image, u, descentDir, step=1, minStep=1e-6):
    
    V = VectorFunctionSpace(mesh, "CG", 1)
    
    delta_u = Function(V)
    new_u = Function(V)
    delta_u.vector()[:] = 0  

    # Compute the functional int_I at the current mesh state
    old_int_I = int_I(mesh, image)
    new_int_I = old_int_I + 1  # Set to ensure the loop runs at least once

    # Start Armijo backtracking with an initial step size
    step = step * 2
    while new_int_I > old_int_I:
        step = step / 2  

        delta_u.vector()[:] = -delta_u.vector()[:]
        ALE.move(mesh, (delta_u))                           # revert change

        # Apply the new deformation direction with the step size
        delta_u.vector()[:] = step * descentDir
        # Deform the mesh (apply the deformation to the mesh)
        ALE.move(mesh, delta_u)
        
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


def update_GD_3D(mesh, mesh_0, image, u, descentDir, step=1, minStep=1e-6):
    
    V = VectorFunctionSpace(mesh, "CG", 1)
    
    delta_u = Function(V)
    new_u = Function(V)
    delta_u.vector()[:] = 0  

    # Compute the functional int_I at the current mesh state
    old_int_I = int_I_3D(mesh, image)
    new_int_I = old_int_I + 1  # Set to ensure the loop runs at least once

    # Start Armijo backtracking with an initial step size
    step = step * 2
    while new_int_I > old_int_I:
        step = step / 2  

        delta_u.vector()[:] = -delta_u.vector()[:]
        ALE.move(mesh, (delta_u))                           # revert change

        # Apply the new deformation direction with the step size
        delta_u.vector()[:] = step * descentDir
        # Deform the mesh (apply the deformation to the mesh)
        ALE.move(mesh, delta_u)
        
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