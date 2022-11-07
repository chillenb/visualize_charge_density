#!/usr/bin/env python3

import argparse
import json
import numpy as np
from scipy.spatial import distance_matrix
import math
from ast import literal_eval as make_tuple

from numba import njit, prange, jit, guvectorize
from numba import float64 as ndbl, boolean as nbool

def uniquetol_old(X, tol=1e-8):
    """Return boolean array indicating the unique rows of X (within a tolerance)"""
    X = np.asarray(X)
    dmat = distance_matrix(X, X, p=np.inf)
    res = np.zeros([X.shape[0]], dtype=bool)
    numba_uniquetol_old(dmat, tol, res)
    return res

def uniquetol(X, tol=1e-8):
    """Return boolean array indicating the unique rows of X (within a tolerance)"""
    X = np.asarray(X)
    res = np.zeros([X.shape[0]], dtype=bool)
    numba_uniquetol(X, tol, res)
    return res

def within_tol(u, v, tol=1e-8):
    return np.linalg.norm(np.asarray(u) - np.asarray(v)) < tol


@njit
def numba_uniquetol(X, tol, res):
    for i in range(X.shape[0]):
        res[i] = True
        for j in range(i):
            if res[j]:
                # row j is unique, so check if
                # row i is different from row j
                stop_because_unique = False
                for k in range(X.shape[1]):
                    if abs(X[i, k] - X[j, k]) > tol:
                        # definitely unique
                        stop_because_unique = True
                        break
                if not stop_because_unique:
                    res[i] = False
                    break

class ChargeDensityEval:
    def __init__(self, shells, density):
        self.shells = shells
        self.density = density

        self.maxam = max([shell["am"] for shell in self.shells])
        self.nbf = density.shape[0]
        self.nprim = sum([shell["nprim"] * (((shell["am"] + 1) * (shell["am"] + 2)) // 2) for shell in self.shells])

        shell_nprim = []
        shell_am = []
        shell_coords = []
        prim_coefs = []
        prim_exps = []

        for shell in shells:
            am = shell["am"]
            shell_nprim.append(shell["nprim"])
            shell_am.append(shell["am"])
            shell_coords.append(shell["coords"])
            for p in range(shell["nprim"]):
                prim_coefs.append(shell["coefs"][p])
                prim_exps.append(shell["exps"][p])
        

        # calculate the number of unique primitive pair centers
        # this eliminates symmetry
        prim_pair_centers = []
        nshells = len(self.shells)
        for i in range(nshells):
            coord1 = np.asarray(shell_coords[i])
            for j in range(i):
                coord2 = np.asarray(shell_coords[j])
                if not within_tol(coord1, coord2):
                    for p1 in range(shell_nprim[i]):
                        for p2 in range(shell_nprim[j]):
                            a = prim_exps[p1]
                            b = prim_exps[p2]
                            p = a + b
                            P = (a * coord1 + b * coord2) / p
                            prim_pair_centers.append(list(P))
        prim_pair_centers.extend(shell_coords)
        self.prim_pair_centers = np.asarray(prim_pair_centers)
        unique_ctrs = uniquetol(self.prim_pair_centers)
        self.unique_prim_pair_centers = self.prim_pair_centers[unique_ctrs]


        self.shell_coords = np.asarray(shell_coords)
        self.shell_nprim = np.asarray(shell_nprim)
        self.shell_am = np.asarray(shell_am)
        self.prim_coefs = np.asarray(prim_coefs)
        self.prim_exps = np.asarray(prim_exps)
        self.natoms = uniquetol(self.shell_coords).sum()
    
    def print_out(self):
        formatstr = '{:<35}{:>10}'
        print(formatstr.format("Number of atoms: ", self.natoms))
        print(formatstr.format("Number of shells: ", len(self.shells)))
        print(formatstr.format("Number of basis functions: ", self.nbf))
        print(formatstr.format("Number of primitives: ", self.nprim))
        print(formatstr.format("Primitive pairs: ", math.comb(self.nprim, 2)))
        print(formatstr.format("Unique primitive pair centers: ", self.unique_prim_pair_centers.shape[0]))

    def compute_phi(self, x, y, z):
        res = np.zeros([x.shape[0], self.nbf])
        compute_phi_numba(
            x, y, z, self.shell_am, self.shell_nprim, self.shell_coords, self.prim_coefs, self.prim_exps, res
        )
        return res

    def compute_density(self, x, y, z):
        chunk_size = 1000000
        res = np.zeros([x.shape[0]])
        buffer = np.zeros([chunk_size, self.nbf])
        compute_density_numba(
            x,
            y,
            z,
            self.shell_am,
            self.shell_nprim,
            self.shell_coords,
            self.prim_coefs,
            self.prim_exps,
            self.density,
            buffer,
            res,
        )
        return res

    def calc_drawbox(self):
        l = [0, 0, 0]
        u = [0, 0, 0]

        for shell in self.shells:
            for i in range(3):
                l[i] = min(l[i], shell["coords"][i])
                u[i] = max(u[i], shell["coords"][i])

        # buffer of 5 bohr
        self.l = [l[i] - 5 for i in range(3)]
        self.u = [u[i] + 5 for i in range(3)]

    def get_grid(self, spacing):
        self.calc_drawbox()
        x = np.arange(self.l[0], self.u[0], spacing)
        y = np.arange(self.l[1], self.u[1], spacing)
        z = np.arange(self.l[2], self.u[2], spacing)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z)
        self.spacing = spacing

    def get_2d_grid(self, spacing, origin, xdir, ydir):
        self.calc_drawbox()
        normal = np.cross(xdir, ydir)
        origin = np.asarray(origin)
        projection = self.shell_coords - origin
        projection = projection - np.outer(np.dot(projection, normal), normal)
        projections_x = np.dot(projection, xdir)
        projections_y = np.dot(projection, ydir)
        l = [np.min(projections_x) - 2, np.min(projections_y) - 2]
        u = [np.max(projections_x) + 2, np.max(projections_y) + 2]

        untransformed_x = np.arange(l[0], u[0], spacing)
        untransformed_y = np.arange(l[1], u[1], spacing)
        self.X, self.Y = np.meshgrid(untransformed_x, untransformed_y)
        A = np.column_stack([xdir, ydir])
        self.slice = (np.column_stack([self.X.flatten(), self.Y.flatten()]) @ A.T) + origin

    def calc_density_on_2dgrid(self):
        self.density_grid = self.compute_density(self.slice[:, 0], self.slice[:, 1], self.slice[:, 2]).reshape(
            self.X.shape
        )

    def calc_density_on_grid(self):
        self.density_grid = self.compute_density(self.X.flatten(), self.Y.flatten(), self.Z.flatten()).reshape(
            self.X.shape
        )
    
    def get_nice_basis_functions(self):
        prim_offset = 0
        ao = 0
        self.nice_basis_functions = []
        for ns in range(len(self.shells)):
            am = self.shell_am[ns]
            nprim = self.shell_nprim[ns]
            center = self.shell_coords[ns]
            coefs = self.prim_coefs[prim_offset:prim_offset + nprim]
            exps = self.prim_exps[prim_offset:prim_offset + nprim]
            for amx in range(am, -1, -1):
                for amy in range(am - amx, -1, -1):
                    amz = am - amx - amy
                    self.nice_basis_functions.append((center, coefs, exps, amx, amy, amz))
                    ao += 1
            prim_offset += nprim



@njit(parallel=True)
def compute_phi_numba(x, y, z, shell_am, shell_nprim, shell_coords, prim_coefs, prim_exps, res):
    npts = x.shape[0]
    for npt in prange(npts):
        prim_offset = 0
        ao = 0
        for ns in range(shell_nprim.shape[0]):
            am = shell_am[ns]
            dx = x[npt] - shell_coords[ns, 0]
            dy = y[npt] - shell_coords[ns, 1]
            dz = z[npt] - shell_coords[ns, 2]
            r2 = dx**2 + dy**2 + dz**2
            cexpr = 0.0
            for p in range(shell_nprim[ns]):
                cexpr += prim_coefs[prim_offset + p] * np.exp((-1.0 * prim_exps[prim_offset + p] * r2))
            for amx in range(am, -1, -1):
                for amy in range(am - amx, -1, -1):
                    amz = am - amx - amy
                    res[npt, ao] = cexpr * dx**amx * dy**amy * dz**amz
                    ao += 1
            prim_offset += shell_nprim[ns]
    return res

@njit(parallel=True)
def compute_single_phi_numba(x, y, z, center, prim_coefs, prim_exps, amx, amy, amz, res):
    npts = x.shape[0]
    for npt in prange(npts):
        dx = x[npt] - center[0]
        dy = y[npt] - center[1]
        dz = z[npt] - center[2]
        r2 = dx**2 + dy**2 + dz**2
        cexpr = 0.0
        for p in range(prim_coefs.shape[0]):
            cexpr += prim_coefs[p] * np.exp((-1.0 * prim_exps[p] * r2))
        res[npt] = cexpr * dx**amx * dy**amy * dz**amz
    return res


@njit(parallel=True)
def compute_density_numba(x, y, z, shell_am, shell_nprim, shell_coords, prim_coefs, prim_exps, density, buffer, res):
    chunk_size = buffer.shape[0]
    npts = x.shape[0]
    for chunk in range(0, npts, chunk_size):
        chunk_lower = chunk
        chunk_upper = min(chunk + chunk_size, npts)
        compute_phi_numba(
            x[chunk_lower:chunk_upper],
            y[chunk_lower:chunk_upper],
            z[chunk_lower:chunk_upper],
            shell_am,
            shell_nprim,
            shell_coords,
            prim_coefs,
            prim_exps,
            buffer,
        )
        for npt in prange(chunk_lower, chunk_upper):
            res[npt] = buffer[npt - chunk_lower, :] @ density @ buffer[npt - chunk_lower, :].T


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
        "-s", "--spacing", type=float, default=0.07, help="spacing of grid points for interactive 3D plot, in Bohr"
    )
    parser.add_argument(
        "--volumeplot",
        action="store_true",
        help="do volume rendering of the density rather than isosurface plot (interactive mode)",
    )
    #parser.add_argument(
    #    "--aopair_radii",
    #    action="store_true",
    #    help="calculate distribution of AO-pair radii",
    #)
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
    plt.title("$\log_{10}$(density)")
    plt.contourf(cde.X, cde.Y, np.log10(cde.density_grid), levels=np.arange(-1.5, 1, 0.1))
    plt.colorbar()
    plt.show()


def interactive(cde, spacing, args):
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
