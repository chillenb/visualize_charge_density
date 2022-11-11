#!/usr/bin/env python3

import argparse
import json
import numpy as np
import math

from chargedensityeval import ChargeDensityEval

def main():
    parser = argparse.ArgumentParser(
        prog="visualize_charge_density",
        description="Plot the electron density of a molecule given a basis set and a density matrix",
    )
    parser.add_argument("molecule", metavar="MOLECULE", help="molecule and basis set (as .json)")
    parser.add_argument("density", metavar="DENSITY", help="density matrix (as .npy)")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="do an interactive 3d plot with pyvista rather than 2d contour plot",
    )
    parser.add_argument(
        "-s",
        "--spacing",
        type=float,
        default=0.07,
        help="spacing of grid points for interactive 3D plot, in Bohr",
    )
    parser.add_argument(
        "--volumeplot",
        action="store_true",
        help="do volume rendering of the density rather than isosurface plot (interactive mode)",
    )
    parser.add_argument(
        "--plane_origin",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        metavar=("o1", "o2", "o3"),
        help="plane origin for 2d contour plot",
    )
    parser.add_argument(
        "--plane_x",
        type=float,
        nargs=3,
        default=[1, 0, 0],
        metavar=("x1", "x2", "x3"),
        help="plane x-direction for 2d contour plot",
    )
    parser.add_argument(
        "--plane_y",
        type=float,
        nargs=3,
        default=[0, 1, 0],
        metavar=("y1", "y2", "y3"),
        help="plane y-direction for 2d contour plot",
    )

    args = parser.parse_args()

    with open(args.molecule) as f:
        shells = json.load(f)

    density = np.load(args.density)

    cde = ChargeDensityEval(shells, density)

    cde.print_out()

    if args.interactive:
        interactive(cde, args.spacing, args)
    else:
        contourplot(cde, args)


def contourplot(cde, args):
    """Make a 2D contour plot of the density

    Args:
        cde (ChargeDensityEval): charge density evaluation object
        args (Namespace): arguments from argparse
    """    
    import matplotlib.pyplot as plt

    xdir = np.asarray(args.plane_x)
    xdir = xdir / np.linalg.norm(xdir)
    ydir = np.asarray(args.plane_y)
    ydir = ydir - ydir.dot(xdir) * xdir
    ydir = ydir / np.linalg.norm(ydir)
    cde.get_2d_grid(0.05, args.plane_origin, xdir, ydir)
    cde.calc_density_on_2dgrid()
    plt.figure()
    plt.xlabel("x (Bohr)")
    plt.ylabel("y (Bohr)")
    plt.title("$\\log_{10}$(density)")
    plt.contourf(cde.X2d, cde.Y2d, np.log10(cde.density_grid), levels=np.arange(-1.5, 1, 0.1))
    plt.colorbar()
    plt.show()


def interactive(cde, spacing, args):
    """
    This function is a modified version of add_mesh_isovalue from pyvista
    (file pyvista/plotting/widgets.py)
    which is licensed under the MIT license (reproduced below).   
    
    The MIT License

    Copyright (c) 2017-2022 The PyVista Developers

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
    """
    import pyvista as pv
    from pyvista import _vtk

    cde.get_grid(spacing)
    cde.calc_density_on_grid()
    pdata = pv.wrap(cde.density_grid)

    grid = pv.UniformGrid()
    grid.dimensions = np.asarray(cde.density_grid.shape)
    grid.origin = tuple(cde.l)
    grid.spacing = [spacing, spacing, spacing]
    grid.point_data["density"] = cde.density_grid.flatten(order="F")
    p = pv.Plotter()
    p.set_background("white")

    if args.volumeplot:
        grid.point_data["density"] = cde.density_grid.flatten(order="F")
        p.add_volume(grid, cmap="viridis", clim=[0, 1], opacity=[0, 1, 1, 1, 1, 1])

    else:
        # isosurface
        alg = _vtk.vtkContourFilter()
        alg.SetInputDataObject(grid)
        alg.SetComputeNormals(True)
        alg.SetComputeGradients(False)
        alg.SetComputeScalars(True)
        field, scalars = grid.active_scalars_info
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars)
        alg.SetNumberOfContours(1)
        isovalue_mesh = pv.wrap(alg.GetOutput())
        p.isovalue_meshes.append(isovalue_mesh)

        def callback(value):
            alg.SetValue(0, 10**value)
            alg.Update()
            isovalue_mesh.shallow_copy(alg.GetOutput())

        p.add_slider_widget(
            callback=callback,
            rng=[-3, 2],
            value=0,
            title="isosurface: log10(density)",
            color="black",
        )
        p.add_mesh(
            isovalue_mesh,
            scalars=scalars,
            cmap="plasma",
            specular=5,
            smooth_shading=True,
            show_scalar_bar=False,
            opacity=0.9,
        )
    p.show()


if __name__ == "__main__":
    main()
