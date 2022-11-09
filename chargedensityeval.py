import numpy as np
import numpy.typing as npt
import math

from numba import njit, prange, jit, guvectorize
from numba import float64 as ndbl, boolean as nbool


class ChargeDensityEval:
    def __init__(self, shells: list[dict], density: np.ndarray):
        self.shells = shells
        self.density = density

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
        """Print basic information about the molecule
        """        
        formatstr = "{:<35}{:>10}"
        print(formatstr.format("Number of atoms: ", self.natoms))
        print(formatstr.format("Number of shells: ", len(self.shells)))
        print(formatstr.format("Number of basis functions: ", self.nbf))
        print(formatstr.format("Number of primitives: ", self.nprim))
        print(formatstr.format("Primitive pairs: ", math.comb(self.nprim, 2)))
        print(
            formatstr.format(
                "Unique primitive pair centers: ",
                self.unique_prim_pair_centers.shape[0],
            )
        )

    def compute_phi(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Compute the value of all basis functions at the given points

        Args:
            x (np.ndarray): x coordinates, array of shape (N,)
            y (np.ndarray): y coordinates, array of shape (N,)
            z (np.ndarray): z coordinates, array of shape (N,)

        Returns:
            np.ndarray: array of shape (N, self.nbf)
        """
        res = np.zeros([x.shape[0], self.nbf])
        compute_phi_numba(
            x,
            y,
            z,
            self.shell_am,
            self.shell_nprim,
            self.shell_coords,
            self.prim_coefs,
            self.prim_exps,
            res,
        )
        return res

    def compute_density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Compute the electron density at a set of points

        Args:
            x (np.ndarray): x coordinates
            y (np.ndarray): y coordinates
            z (np.ndarray): z coordinates

        Returns:
            np.ndarray: electron density evaluated at the given points
        """
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
        """
        Region to plot is just a bounding box of the atom
        coordinates with 5 Bohr of buffer on each side.
        """
        l = [0, 0, 0]
        u = [0, 0, 0]

        for shell in self.shells:
            for i in range(3):
                l[i] = min(l[i], shell["coords"][i])
                u[i] = max(u[i], shell["coords"][i])

        # buffer of 5 bohr
        self.l = [l[i] - 5 for i in range(3)]
        self.u = [u[i] + 5 for i in range(3)]

    def get_grid(self, spacing: float):
        """Get a 3D grid of points for plotting

        Args:
            spacing (float): 1D spacing between points
        """
        self.calc_drawbox()
        x = np.arange(self.l[0], self.u[0], spacing)
        y = np.arange(self.l[1], self.u[1], spacing)
        z = np.arange(self.l[2], self.u[2], spacing)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z)
        self.spacing = spacing

    def get_2d_grid(self, spacing: float, origin: np.ndarray, xdir: np.ndarray, ydir: np.ndarray):
        """Get a 2D grid of points on a plane for plotting

        Args:
            spacing (float): 1D spacing between points
            origin (np.ndarray): origin of the plane
            xdir (np.ndarray): x direction on the plane
            ydir (np.ndarray): y direction on the plane
        """
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
        self.X2d, self.Y2d = np.meshgrid(untransformed_x, untransformed_y)
        A = np.column_stack([xdir, ydir])
        self.slice = (np.column_stack([self.X2d.flatten(), self.Y2d.flatten()]) @ A.T) + origin

    def calc_density_on_2dgrid(self):
        """Calculate the electron density on a 2D grid
        """
        self.density_grid = self.compute_density(self.slice[:, 0], self.slice[:, 1], self.slice[:, 2]).reshape(
            self.X2d.shape
        )

    def calc_density_on_grid(self):
        """Calculate the electron density on a 3D grid
        """
        self.density_grid = self.compute_density(self.X.flatten(), self.Y.flatten(), self.Z.flatten()).reshape(
            self.X.shape
        )

    def get_nice_basis_functions(self):
        """

        :return:
        """
        prim_offset = 0
        ao = 0
        self.nice_basis_functions = []
        for ns in range(len(self.shells)):
            am = self.shell_am[ns]
            nprim = self.shell_nprim[ns]
            center = self.shell_coords[ns]
            coefs = self.prim_coefs[prim_offset : prim_offset + nprim]
            exps = self.prim_exps[prim_offset : prim_offset + nprim]
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
            accum = 0.0
            for p in range(shell_nprim[ns]):
                accum += prim_coefs[prim_offset + p] * np.exp((-1.0 * prim_exps[prim_offset + p] * r2))
            for amx in range(am, -1, -1):
                for amy in range(am - amx, -1, -1):
                    amz = am - amx - amy
                    res[npt, ao] = accum * dx**amx * dy**amy * dz**amz
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
        accum = 0.0
        for p in range(prim_coefs.shape[0]):
            accum += prim_coefs[p] * np.exp((-1.0 * prim_exps[p] * r2))
        res[npt] = accum * dx**amx * dy**amy * dz**amz
    return res


@njit(parallel=True)
def compute_density_numba(
    x,
    y,
    z,
    shell_am,
    shell_nprim,
    shell_coords,
    prim_coefs,
    prim_exps,
    density,
    buffer,
    res,
):
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



def uniquetol(X: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Return boolean array indicating the unique rows of X (within a tolerance)

    Args:
        X (np.ndarray): array of shape (N, M)
        tol (float, optional): tolerance. Defaults to 1e-8.

    Returns:
        np.ndarray: boolean array of shape (N,)
    """
    X = np.asarray(X)
    res = np.zeros([X.shape[0]], dtype=bool)
    numba_uniquetol(X, tol, res)
    return res


@njit
def numba_uniquetol(X: np.ndarray, tol: float, res: np.ndarray):
    """Numba implementation of uniquetol"""
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


def within_tol(u: npt.ArrayLike, v: npt.ArrayLike, tol=1e-8) -> bool:
    """Return true if u and v are close in L2 norm

    Args:
        u (np.ndarray): vector
        v (np.ndarray): vector
        tol (float, optional): tolerance. Defaults to 1e-8.

    Returns:
        bool: whether u and v are close in L2 norm
    """
    return np.linalg.norm(np.asarray(u) - np.asarray(v)) < tol

