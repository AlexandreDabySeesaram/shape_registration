import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import dolfin
import dolfin_warp as dwarp
import mshr
import dolfin_mech as dmech
from shape_derivatives import *

def read_vti_to_numpy(file_path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    image_data = reader.GetOutput()

    dims = image_data.GetDimensions()
    print(f"Image Dimensions: {dims}")

    point_data = image_data.GetPointData()
    if point_data.GetNumberOfArrays() == 0:
        raise ValueError("No point data found in the VTI file.")

    array = point_data.GetArray(0)
    if array is None:
        raise ValueError("No valid data array found in the VTI file.")
    numpy_array = vtk_to_numpy(array)
    numpy_array = numpy_array.reshape(dims)  
    return numpy_array


#%% Domain omega generation

sphere_center = (120, 120, 120)                                         # Center of the sphere
sphere_radius = 75                                                     # Radius of the sphere

resolution = 30                                                         # Resolution of the mesh

# Create a 3D spherical domain
center = dolfin.Point(sphere_center[0], sphere_center[1], sphere_center[2])              # Center of the disc
radius = 1.4*sphere_radius                                      # Radius of the disc
domain = mshr.Sphere(center, radius)
mesh_omega = mshr.generate_mesh(domain, resolution)
mesh_omega.num_vertices()

#%% Image expression generation

Python_reader = False
dwarp_reader  = True

# Naive python loops

image_path = "PA5_Binary_signed_int.vti" 




class FE_image_3D(dolfin.UserExpression):

    def eval_cell(self, value, x, ufc_cell):
        p = dolfin.Cell(mesh_cube, dolfin.ufc_cell.index).midpoint()
        i, j, k = int(p[0]*(Nx-1)), int(p[1]*(Ny-1)), int(p[2]*(Nz-1))
        value[:] = voxel_array[-(j+1), i, k]

    def value_shape(self):
        return ()

if Python_reader:
    voxel_array = read_vti_to_numpy(image_path)
    [Nx, Ny, Nz] = voxel_array.shape
    print(f"Voxel Data Shape: {voxel_array.shape}")
    mesh_cube = dolfin.UnitCubeMesh(voxel_array.shape[0], voxel_array.shape[1], voxel_array.shape[2])
    image_expr = FE_image_3D()


# Expression cpp

if dwarp_reader:

    fe = dolfin.FiniteElement(
        family="Quadrature",
        cell=mesh_omega.ufl_cell(),
        quad_scheme="default")
    name, cpp = dwarp.get_ExprIm_cpp_pybind(
        im_dim=3,
        im_type="im",
        verbose=0)
    module = dolfin.compile_cpp_code(cpp)
    expr = getattr(module, name)
    Img_3D_expr = dolfin.CompiledExpression(
        expr(),
        element=fe)
    Img_3D_expr.init_image(
        filename=image_path)



#%% Tracking



V_fs = dolfin.FunctionSpace(mesh_omega, "CG", 1)

i_proj = dolfin.interpolate(Img_3D_expr, V_fs)

# Img_3D_expr = dolfin.Expression("f - 1", degree=1, f=Img_3D_expr)
dmech.write_VTU_file(
    filebasename="I_sphere_3",
    function=I_3D(mesh_omega, Img_3D_expr),
    time=0,
    preserve_connectivity=True
)



#%% Naive gradient descent


# Algorithms parameters
maxit = 500           # max number of iteration
step = 0.01            # initial step size
coeffStep = 1.5       # step increase factor at each iteration ( > 1)
minStep = 1e-9        # minimum step size (stop criterion)

# Shape derivative parameters
alpha = 1            # dissipation term
gamma = 1            # preserve mesh quality (arrondis aussi les angles...)

# Initialization

# mesh_Omega_0 = Mesh(mesh_omega)
# u_fs_Omega_0 = VectorFunctionSpace(mesh_Omega_0, "CG", 1)
# u_Omega_0 = Function(u_fs_Omega_0, name="mapping")


u_fs = dolfin.VectorFunctionSpace(mesh_omega, "CG", 1)
u = dolfin.Function(u_fs, name="mapping")
u_test = dolfin.TestFunction(u_fs)
u.vector()[:] = 0
dolfin.ALE.move(mesh_omega, u)


### MARTIN DEBUG
# dV = dolfin.Measure("dx", domain=mesh_Omega_0)
# F = dolfin.Identity(3) + dolfin.grad(u)
# J = dolfin.det(F)

# Psi = Img_3D_expr * J

# dolfin.assemble(Psi * dV)

# dPsi = dolfin.derivative(Psi, u, u_test)
# dPsi += dolfin.inner(GradImg_3D_expr, u_test) * J

# dolfin.assemble(dPsi * dV) # Replace solving the bilinear problem by choosing L2 inner product then add mech regularisation
### MARTIN DEBUG





loss_vect = [int_I_3D(mesh_omega, Img_3D_expr)]

# Optimization loop (gradient descent)
k = 0

dmech.write_VTU_file(
filebasename = "mapping_lung_3D",
function = u_Omega_0,
time = k,
preserve_connectivity = True)

while k<maxit and step >= minStep:
    k += 1
    # shape derivative computation and update
    shape_gradient = shape_derivative_volume(mesh_omega, u, I_3D(mesh_omega, Img_3D_expr), grad_I_3D(mesh_omega, Img_3D_expr), alpha = alpha, gamma = gamma)
    u, loss , step = update_GD_3D(mesh_omega, Img_3D_expr, u, -shape_gradient, step = step * coeffStep, minStep = minStep)

    u_Omega_0.vector()[:] = u.vector()[:]
    # Print and store result
    print(f"it = {k}  |  loss = {loss:.10e}    ", end = "\r")
    loss_vect.append(loss)

    dmech.write_VTU_file(
    filebasename = "mapping_lung_3D",
    function = u_Omega_0,
    time = k,
    preserve_connectivity = True)





