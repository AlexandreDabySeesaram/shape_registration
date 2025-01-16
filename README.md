# This code allows to register 2 and 3D shapes

## Getting started

### 2D registration
for the 2D case only dolfin_mech and vtk are required so the required environment can be set up as follow

```
conda create -y -c conda-forge -n shape_registration-env expat=2.5 fenics=2019.1.0 gnuplot=5.4 matplotlib=3.5 meshio=5.3 mpi4py=3.1.3 numpy=1.23.5 pandas=1.3 pip python=3.10 scipy=1.9 vtk=9.2
conda activate shape_registration
conda env config vars set CPATH=$CONDA_PREFIX/include/vtk-9.2
conda activate shape_registration
pip install dolfin_mech
```

### 3D registration

In addition to the 2D requirements, the 3D registration requires ```dolfin_warp``` in it's unstable development. It therefore needs to be installed in `editable` mode as follows

``````
conda activate shape_registration
git clone git@gitlab.inria.fr:mgenet/dolfin_warp.git
cd dolfin_warp
pip install -e .
git checkout devel-Alexan…BlurTracking
``````

### Default folder set-up


```
.
|
|____README.md
|
|____pipeline_2D.py
|____pipeline_3D.py
|____shape_derivatives.py
|____create_data.py
|____Images
| |
| |____your_image.vti
|
|____Results
| |
| |____your_results
```