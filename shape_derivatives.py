import dolfin
import ufl
import numpy as np

#%% Define projection of the image onto the domain mesh


def proj_I(
    mesh                : dolfin.Mesh           = None,            
    image_expression    : dolfin.Expression     = None):
    """
    i_proj = proj_I(mesh, image_expression)
    
    returns the projection of image_expression onto mesh
    """

    V       = dolfin.FunctionSpace(mesh, "CG", 1)
    i_proj  = dolfin.interpolate(image_expression, V)
    
    return i_proj

def grad_I(
    mesh                : dolfin.Mesh           = None,            
    image_expression    : dolfin.Expression     = None):
    """
    grad_i_proj = grad_I(mesh, image_expression)
    
    returns the gradient of the projection of image_expression onto mesh
    """

    V       = dolfin.FunctionSpace(mesh, "CG", 1)
    i_proj  = dolfin.interpolate(image_expression, V)
    
    return dolfin.grad(i_proj)


def int_I(
    mesh                : dolfin.Mesh           = None,            
    image_expression    : dolfin.Expression     = None):
    """
    int_I = int_I(mesh, image_expression)
    
    returns the integral of the projection of image_expression onto mesh
    """
    i_proj = proj_I(mesh, image_expression)

    dV = dolfin.Measure(
            "dx",
            domain=mesh)
    # print(f"using expression integral is:{dolfin.assemble(image_expression*dV)}")#DEBUG
    # print(f"using projection integral is:{dolfin.assemble(i_proj*dV)}")#DEBUG
    
    return dolfin.assemble(image_expression*dV)
    # return dolfin.assemble(i_proj * dolfin.dx)



def shape_derivative_volume(mesh, I, grad_I, alpha=1):
    V                   = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    u, v                = dolfin.TrialFunction(V), dolfin.TestFunction(V) 
    shape_derivative    = dolfin.div(v) * I * dolfin.dx + dolfin.inner(grad_I, v) * dolfin.dx
    # shape_derivative    = dolfin.div(v) * dolfin.dx #DEBUG False, for comparison
    # shape_derivative    = dolfin.div(v) * I * dolfin.dx #DEBUG False, for comparison with image
    
    
    dV = dolfin.Measure(
            "dx",
            domain=mesh)
    print(dolfin.assemble(I*dV))#DEBUG
    # Regularization term (choice of inner_product)
    inner_product_1       = dolfin.inner(dolfin.grad(u) + dolfin.grad(u).T, dolfin.grad(v) + dolfin.grad(v).T) * dolfin.dx 
    inner_product_2       = alpha * dolfin.inner(u, v) * dolfin.dx
    inner_product         = inner_product_1 + inner_product_2


    # Solve the system to find shape_gradient
    shape_gradient      = dolfin.Function(V)
    # dolfin.solve(inner_product == shape_derivative, shape_gradient)


    inner_assembled = dolfin.assemble(inner_product)
    rhs_assembled = dolfin.assemble(shape_derivative)

    dolfin.solve(inner_assembled,shape_gradient.vector(),rhs_assembled)

    return shape_gradient.vector()[:]

 


def check_flipped_triangles(mesh):

    det_J = np.zeros(mesh.num_cells())
    for i, cell in enumerate(dolfin.cells(mesh)):
        # Compute the Jacobian determinant of the cell
        det_J_i     = cell.volume()
        det_J[i]    = det_J_i
        if det_J_i < 0:
            print("negative jac")
            return 1
    negative_count = np.sum(det_J < 0)

    return negative_count>0





def update_GD(mesh, image, u, descentDir, step=1, minStep=1e-6):
    
    V           = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    delta_u     = dolfin.Function(V)
    new_u       = dolfin.Function(V)
    delta_u.vector()[:] = 0  

    # Compute the functional int_I at the current mesh state
    old_int_I   = int_I(mesh, image)
    new_int_I   = old_int_I + 1  
    k = 1
    # Start Armijo backtracking with an initial step size
    step = step * 2
    while new_int_I > old_int_I:
    # for i in range(1):
        step = step / 2  
        # step = 1#DEBUG comparison
        # If backtracking required (new loop), revert changes to the mesh before trying a new step
        delta_u.vector()[:] = -delta_u.vector()[:]
        dolfin.ALE.move(mesh, (delta_u))                           

        delta_u.vector()[:] = step * descentDir
        dolfin.ALE.move(mesh, delta_u)
        
        new_u.vector()[:] = u.vector()[:] + delta_u.vector()[:]

        # Check for flipped triangles (you would need to implement your own logic here)
        n_flipped = check_flipped_triangles(mesh)
        if n_flipped > 0:
            print(f"* {n_flipped} flipped triangles found, decreasing step size")
            continue  
        
        new_int_I = int_I(mesh, image)
        print(f"*** old int I : {old_int_I:.4e}, new int I {new_int_I:.4e}, relax {step:.4e}")
        if step <= minStep:
            print(f"Backtracking failed, step <= {minStep}")
            break
        print(f"    relaxation iteration: {k}")
        k+=1
    return new_u, new_int_I, step