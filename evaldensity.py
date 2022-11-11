#!/usr/bin/env python3

import argparse
import json
import numpy as np

from chargedensityeval import ChargeDensityEval


def main():
    parser = argparse.ArgumentParser(
        prog="eval_density",
        description="Evaluate the electron density of a molecule on a 2D grid, given a basis set and a density matrix",
    )
    parser.add_argument(
        "molecule", metavar="MOLECULE", help="molecule and basis set (as .json)"
    )
    parser.add_argument("density", metavar="DENSITY", help="density matrix (as .npy)")
    parser.add_argument(
        "-s",
        "--spacing",
        type=float,
        default=0.05,
        help="spacing of grid points for 2D plot, in Bohr",
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
    parser.add_argument(
        "-o", "--output", type=str, default="density_slice.npz", help="output file name"
    )

    args = parser.parse_args()

    with open(args.molecule) as f:
        shells = json.load(f)

    density = np.load(args.density)

    cde = ChargeDensityEval(shells, density)

    cde.print_out()
    eval_on_2d_grid(cde, args)


def eval_on_2d_grid(cde, args):
    """Evaluate the density on a 2D grid, and save the result to an npz file.

    Args:
        cde (ChargeDensityEval): charge density evaluation object
        args (Namespace): arguments from argparse
    """
    xdir = np.asarray(args.plane_x)
    xdir = xdir / np.linalg.norm(xdir)
    ydir = np.asarray(args.plane_y)
    ydir = ydir - ydir.dot(xdir) * xdir
    ydir = ydir / np.linalg.norm(ydir)
    cde.get_2d_grid(args.spacing, args.plane_origin, xdir, ydir)
    cde.calc_density_on_2dgrid()
    np.savez(
        args.output,
        x=cde.X2d,
        y=cde.Y2d,
        density=cde.density_grid,
        points=np.resize(cde.slice, cde.X2d.shape + (3,))
    )

    import IPython

    IPython.embed()
    # You can load and plot this data with:
    # npzfile = np.load(filename)
    # x = npzfile['x']
    # y = npzfile['y']
    # x and y are plane coordinates
    # npzfile['points'] are the corresponding points in 3D space
    # density = npzfile['density']
    # plt.contourf(x, y, density)


if __name__ == "__main__":
    main()
