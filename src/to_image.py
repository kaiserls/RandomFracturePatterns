import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import pyvista as pv
from pyvista import examples
import numpy as np


def to_image_pyvista(app: str):
    # Load the structured grid
    grid = pv.UnstructuredGrid(f"data/vtk/{app}.vtk")

    plotter = pv.Plotter(off_screen=True)
    plotter.window_size = [1000, 2000]
    plotter.add_mesh(
        grid, scalars="OP", show_scalar_bar=False, cmap="gray"
    )  # , cmap="jet") #, show_edges=True, show_scalar_bar=True, show_grid=True, notebook=False)
    plotter.view_xy()
    plotter.screenshot(f"data/pictures/{app}_pyvista.png", transparent_background=True)
    img = plotter.show(return_img=True)


def to_image_vtk(app: str):
    colors = vtk.vtkNamedColors()

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(f"data/vtk/{app}.vtk")
    reader.Update()

    data = reader.GetOutput().GetPointData()
    op_data = data.GetArray("OP")
    print(op_data.GetRange())

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(reader.GetOutput())
    mapper.SetScalarRange(op_data.GetRange())
    # mapper.SetScalarModeToUsePointData()
    mapper.ScalarVisibilityOn()
    mapper.SelectColorArray("OP")

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.Render()

    # Create a render window interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renWin)
    interactor.Initialize()
    interactor.Start()

    # w2if = vtk.vtkWindowToImageFilter()
    # w2if.SetInput(renWin)
    # w2if.Update()

    # writer = vtk.vtkPNGWriter()
    # writer.SetFileName(f"data/pictures/{app}_vtk.png")
    # writer.SetInputConnection(w2if.GetOutputPort())
    # writer.Write()

    # table = vtk.vtkScalarsToColors()
    # table.SetRange(0.0, 1.0) # set the range of your data values

    # convert = vtk.vtkImageMapToColors()
    # convert.SetLookupTable(table)
    # convert.SetOutputFormatToRGB()
    # convert.SetInputData()
    # convert.Update()

    # writer = vtk.vtkPNGWriter()
    # writer.SetInputData(convert.GetOutput())


if __name__ == "__main__":
    import os

    vtkfiles_in_app = {}

    vtk_path = "data/vtk_regular/"

    # Find all apps(folders) and contained vtk files in the data/vtk folder
    for app in os.listdir(vtk_path):
        vtkfiles_in_app[app] = []
        for file in os.listdir(vtk_path + app):
            if file.endswith(".vtk"):
                vtkfiles_in_app[app].append(vtk_path + app + "/" + file)

    # Determine max number of vtk files in an app
    max_vtkfiles = 0
    for app in vtkfiles_in_app:
        if len(vtkfiles_in_app[app]) > max_vtkfiles:
            max_vtkfiles = len(vtkfiles_in_app[app])
    n_apps = len(vtkfiles_in_app)

    # scalar = "LAMBDA_S"
    scalar = "OP"
    # Create a plotter with multiple render windows
    plotter: pv.Plotter = pv.Plotter(shape=(n_apps, max_vtkfiles))
    # Iterate over the VTK files
    i_app = 0
    for app, vtk_files in vtkfiles_in_app.items():
        i_vtk = 0
        for vtk_file in vtk_files:
            # Check if the file exists
            if os.path.isfile(vtk_file):
                # Load the VTK file
                mesh = pv.read(vtk_file)
                # Set the active render window
                plotter.subplot(i_app, i_vtk)
                # Add the mesh to the plotter
                plotter.add_mesh(mesh, scalars=scalar)  # , show_edges=True)
                plotter.reset_camera_clipping_range()
                plotter.view_xy()
                if i_vtk == 0:
                    name = app.split("_")[-1]
                    plotter.add_text(name, position="upper_edge", font_size=12)
            else:
                print(f"File {vtk_file} does not exist.")
            i_vtk += 1
        i_app += 1
    # Show the result
    plotter.show()
