
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt


import dolfin_mech as dmech
from shape_derivatives import *

#%% Synthetic image generation

# Parameters for the domain and semicircle
square_size = 1.0  # Length of the sides of the square
circle_center = (0.5, 0.5)  # Center of the circle
circle_radius = 0.25  # Radius of the circle

# Create a square mesh with an included half-circle
class HalfCircleDomain(SubDomain):
    def inside(self, x, on_boundary):
        dx, dy = x[0] - circle_center[0], x[1] - circle_center[1]
        inside_circle = dx**2 + dy**2 <= (circle_radius+0.1)**2
        upper_half = x[1] >= circle_center[1]
        return inside_circle and upper_half

# Define the full domain and mark the subdomain for the half-circle
mesh = RectangleMesh(Point(0, 0), Point(square_size, square_size), 30, 30)
half_circle = HalfCircleDomain()

# Refine mesh near the half-circle boundary
for _ in range(2):  # Refinement levels
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    cell_markers.set_all(False)
    for cell in cells(mesh):
        if half_circle.inside(cell.midpoint().array(), False):
            cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)

# Define the scalar function space
V = FunctionSpace(mesh, 'CG', 1)

# Define the scalar image
image = Function(V, name="pixel intensity")

# Coordinates of the mesh points
coordinates = V.tabulate_dof_coordinates()

def semicircle_condition(x):
    """
    Check if a point x is inside the half-circle.
    The half-circle is the upper half (y >= center_y).
    """
    dx, dy = x[0] - circle_center[0], x[1] - circle_center[1]
    inside_circle = dx**2 + dy**2 <= circle_radius**2
    upper_half = x[1] >= circle_center[1]
    return inside_circle and upper_half

# Define the scalar image values
image_values = np.ones(len(coordinates))
for i, coord in enumerate(coordinates):
    if semicircle_condition(coord):
        image_values[i] = -1

# Assign values to the scalar image
image.vector()[:] = image_values

plot(image, title="Scalar image")
image.set_allow_extrapolation(True)


dmech.write_VTU_file(
    filebasename = "testShape",
    function = image,
    time = 0,
    preserve_connectivity = True)



#%% 

import matplotlib.pyplot as plt
from dolfin import *


img = plt.imread("lung.pgm")
(Nx, Ny) = img.shape
mesh_img = UnitSquareMesh(Nx, Ny, "crossed")



class FE_image_self(UserExpression):

    def eval_cell(self, value, x, ufc_cell):
        p = Cell(mesh_img, ufc_cell.index).midpoint()
        i, j = int(p[0]*(Nx-1)), int(p[1]*(Ny-1))
        value[:] = img[-(j+1), i]

    def value_shape(self):
        return ()

img = FE_image_self()



# Create the scalar function space for the image field
V = FunctionSpace(mesh_img, "Lagrange", 1)

# Create an instance of the FE_image and interpolate onto V
img_interpolated = interpolate(img, V)

dmech.write_VTU_file(
    filebasename = "img_interpolated",
    function = img_interpolated,
    time = 0,
    preserve_connectivity = True)



#%%
# mesh_omega = UnitSquareMesh(100, 100)

# Create a disc mesh
from mshr import *
center = Point(circle_center[0], circle_center[1])  # Center of the disc
radius = 1.4*circle_radius         # Radius of the disc
resolution = 50                   # Resolution of the mesh
domain = Circle(center, radius)
mesh_omega = generate_mesh(domain, resolution)

integral_value = int_I(mesh_omega, image)

print(f"Integral of f over the domain: {integral_value}")

plot(I(mesh_omega, image))

dmech.write_VTU_file(
    filebasename = "I_projected",
    function = I(mesh_omega, image),
    time = 0,
    preserve_connectivity = True)



# %% Naive gradient descent


# Algorithms parameters
maxit = 500           # max number of iteration
step = 0.5            # initial step size
coeffStep = 1.5       # step increase factor at each iteration ( > 1)
minStep = 1e-9        # minimum step size (stop criterion)

# Shape derivative parameters
alpha = 1            # dissipation term
gamma = 1            # preserve mesh quality (arrondis aussi les angles...)

# Initialization

mesh_Omega_0 = Mesh(mesh_omega)
u_fs_Omega_0 = VectorFunctionSpace(mesh_Omega_0, "CG", 1)
u_Omega_0 = Function(u_fs_Omega_0, name="mapping")


u_fs = VectorFunctionSpace(mesh_omega, "CG", 1)
u = Function(u_fs, name="mapping")
u.vector()[:] = 0
ALE.move(mesh_omega, u)


loss_vect = [int_I(mesh_omega, image)]

# Optimization loop (gradient descent)
k = 0

dmech.write_VTU_file(
filebasename = "mapping_Omega_0_bis",
function = u_Omega_0,
time = k,
preserve_connectivity = True)

while k<maxit and step >= minStep:
    k += 1
    # shape derivative computation and update
    shape_gradient = shape_derivative_volume(mesh_omega, u, I(mesh_omega, image), grad_I(mesh_omega, image), alpha = alpha, gamma = gamma)
    u, loss , step = update_GD(mesh_omega, mesh_Omega_0, image, u, -shape_gradient, step = step * coeffStep, minStep = minStep)
    # mesh_omega = Mesh(mesh_Omega_0)
    # ALE.move(mesh_omega, u)
    u_Omega_0.vector()[:] = u.vector()[:]
    # Print and store result
    print(f"it = {k}  |  loss = {loss:.10e}    ", end = "\r")
    loss_vect.append(loss)

    # dmech.write_VTU_file(
    #     filebasename = "omega_sol",
    #     function = I(mesh_omega, image),
    #     time = k,
    #     preserve_connectivity = True)
    dmech.write_VTU_file(
    filebasename = "mapping_Omega_0_bis",
    function = u_Omega_0,
    time = k,
    preserve_connectivity = True)