import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def sign_masking_binary_UINT(input_name, suffix = "signed", field_name = "pixel intensity"):


    input_file                  = input_name+".vti"

    # Load the VTI file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(input_file)
    reader.Update()

    # Get the image data
    image_data = reader.GetOutput()

    # Extract scalar data and convert to Numpy array
    scalars = image_data.GetPointData().GetScalars()
    scalar_array = vtk_to_numpy(scalars)

    # Apply the value adjustments
    scalar_array[(scalar_array == 200)] = 0
    scalar_array[(scalar_array == 0)] = 2
    scalar_array[(scalar_array == 100)] = 0
    scalar_array = scalar_array.astype(np.uint8)


    # Convert the modified Numpy array back to VTK format
    modified_scalars = numpy_to_vtk(scalar_array)
    modified_scalars.SetName(field_name)  # Set the name of the scalar field

    # Assign the modified scalars back to the image data
    image_data.GetPointData().SetScalars(modified_scalars)

    # Save the modified image to a new VTI file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(input_name+"_"+suffix+".vti")
    writer.SetInputData(image_data)
    writer.Write()

    print("Done sign masking. "+input_name)

def sign_masking_binary_INT(input_name, suffix = "signed", field_name = "pixel intensity"):


    input_file                  = input_name+".vti"

    # Load the VTI file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(input_file)
    reader.Update()

    # Get the image data
    image_data = reader.GetOutput()

    # Extract scalar data and convert to Numpy array
    scalars = image_data.GetPointData().GetScalars()
    scalar_array = vtk_to_numpy(scalars)

    # Apply the value adjustments
    scalar_array[(scalar_array == 200)] = 0
    scalar_array[(scalar_array == 0)] = 50
    scalar_array[(scalar_array == 100)] = -50
    scalar_array = scalar_array.astype(np.int8)


    # Convert the modified Numpy array back to VTK format
    modified_scalars = numpy_to_vtk(scalar_array)
    modified_scalars.SetName(field_name)  # Set the name of the scalar field

    # Assign the modified scalars back to the image data
    image_data.GetPointData().SetScalars(modified_scalars)

    # Save the modified image to a new VTI file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(input_name+"_"+suffix+".vti")
    writer.SetInputData(image_data)
    writer.Write()

    print("Done sign masking. "+input_name)

def sign_masking_binary_DOUBLE(input_name, suffix = "signed", field_name = "pixel intensity"):

    input_file                  = input_name+".vti"

    # Load the VTI file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(input_file)
    reader.Update()

    # Get the image data
    image_data = reader.GetOutput()

    # Extract scalar data and convert to Numpy array
    scalars = image_data.GetPointData().GetScalars()
    scalar_array = vtk_to_numpy(scalars)

    # Apply the value adjustments
    scalar_array[(scalar_array == 200)] = 0
    scalar_array[(scalar_array == 0)] = 50
    scalar_array[(scalar_array == 100)] = -50
    scalar_array = scalar_array.astype(np.float64)


    # Convert the modified Numpy array back to VTK format
    modified_scalars = numpy_to_vtk(scalar_array)
    modified_scalars.SetName(field_name)  # Set the name of the scalar field

    # Assign the modified scalars back to the image data
    image_data.GetPointData().SetScalars(modified_scalars)

    # Save the modified image to a new VTI file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(input_name+"_"+suffix+".vti")
    writer.SetInputData(image_data)
    writer.Write()

    print("Done sign masking. "+input_name)