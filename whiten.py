import functools
import glob
import mdtraj as md
import multiprocessing as mp
import numpy as np
import os

from scipy.linalg import inv, sqrtm


def apply_unwhitening(whitened, uwm, cm):
    """Set cm=0 if no center of mass to restore"""
    # multiply each row in whitened by c00_sqrt
    coords = np.einsum('ij,aj->ai', uwm, whitened)
    coords += cm
    return coords


def get_c00(coords, cm):
    """Set cm=0 if center of mass is already removed"""
    coords -= cm
    n_coords = coords.shape[0]
    norm_const = 1.0 / n_coords
    #c00 = np.einsum('bi,bo->io', coords, coords)
    #matrix math this instead
    c00 = np.matmul(coords.transpose(),coords)
    c00 *= norm_const
    return c00


def _get_c00_xtc(xtc_fn, top, cm):
    """Set cm=0 if center of mass is already removed"""
    print(xtc_fn)
    traj = md.load(xtc_fn, top=top)

    n = len(traj)
    n_atoms = traj.top.n_atoms
    coords = traj.xyz.reshape((n, 3 * n_atoms))
    return get_c00(coords, cm), len(traj)


def get_c00_xtc_list(xtc_fns, top, cm, n_cores):
    """Set cm=0 if center of mass is already removed"""
    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_get_c00_xtc, top=top, cm=cm)
    result = pool.map_async(f, xtc_fns)
    result.wait()
    r = result.get()
    pool.close()

    n = len(r)
    c00s = np.zeros((n, r[0][0].shape[0], r[0][0].shape[1]))
    lens = np.zeros(n)
    for i in range(n):
        c00s[i] = r[i][0]
        lens[i] = r[i][1]
    c00 = np.einsum('ijk,i->jk', c00s, lens)
    c00 /= lens.sum()
    return c00


def apply_whitening(coords, wm, cm):
    # multiply each row in coords by inv_c00
    whitened = np.einsum('ij,aj->ai', wm, coords)
    return whitened


def _apply_whitening_xtc_fn(xtc_fn, top, outdir, wm, cm):
    print("whiten", xtc_fn)
    traj = md.load(xtc_fn, top=top)

    n = len(traj)
    n_atoms = traj.top.n_atoms
    coords = traj.xyz.reshape((n, 3 * n_atoms))
    coords -= cm
    whitened = apply_whitening(coords, wm, cm)
    dir, fn = os.path.split(xtc_fn)
    new_fn = os.path.join(outdir, fn)
    traj = md.Trajectory(whitened.reshape((n, n_atoms, 3)), top)
    traj.save(new_fn)


def get_wuw_mats(c00):
    """uwm for unwhitening, wm for whitening matrix"""
    uwm = sqrtm(c00).real
    wm = inv(uwm).real
    return uwm, wm


def apply_whitening_xtc_dir(xtc_dir, top, wm, cm, n_cores, outdir):
    """Set cm=0 if center of mass is already removed"""
    xtc_fns = np.sort(glob.glob(os.path.join(xtc_dir, "*.xtc")))

    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_apply_whitening_xtc_fn, top=top, outdir=outdir, wm=wm, cm=cm)
    pool.map(f, xtc_fns)
    pool.close()
