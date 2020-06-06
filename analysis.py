import mdtraj as md
import nnutils
import multiprocess as mp

class Analysis:

    def __init__(self, net, nav):
        self.net = net
        self.nav = nav
        self.top = md.load(os.path.join(
                           self.nav.whit_data_dir, "master.pdb")
        self.cm = np.load(os.path.join(self.nav.whit_data_dir, "cm.npy")
        self.n_cores = mp.cpu_count()

    def encode_data(self):
        enc_dir = os.path.join(self.nav.net_dir, "encodings")
        nnutils.mkdir(enc_dir)
        xtc_dir = os.path.join(self.nav.whit_data_dir, "aligned_xtcs")
        encode_dir(net, xtc_dir, enc_dir, self.top, self.n_cores, self.cm)
        net.encoder1[-1].out_features

    def recon_traj(self,enc):
        recon_dir = os.path.join(self.nav.net_dir, "recon_trajs")
        nnutils.mkdir(recon_dir)
        enc_dir = os.path.join(self.nav.net_dir, "encodings")
        recon_traj_dir(self.net, enc_dir, recon_dir, self.top,
                       self.cm, self.n_cores)
        print("trajectories reconstructed")

    def get_labels(self,enc):
        label_dir = os.path.join(self.nav.net_dir, "labels")
        nnutils.mkdir(label_dir)
        enc_dir = os.path.join(self.nav.net_dir, "encodings")
        calc_labels(self.net, enc_dir, label_dir, self.n_cores)
        print("labels calculated for all states")

    def get_rmsd(self):
        rmsd_fn = os.path.join(self.nav.net_dir, "rmsd.npy")
        recon_dir = os.path.join(self.nav.net_dir, "recon_trajs")
        orig_xtc_dir = os.path.join(self.nav.whit_data_dir, "aligned_xtcs")
        rmsd = rmsd_dists_dir(recon_dir, orig_xtc_dir, self.top, self.n_cores)
        np.save(rmsd_fn, rmsd)

    def morph():
        pass

    def project_labels():
        pass 

    def check_loss():
        pass

    def clust_enc():
        pass

    def find_feats():
        pass

    def calc_auc():
        pass

    def run_simple(self):
        self.encode_data()
        
        self.recon_traj()

        self.get_labels()
        
        self.get_rmsd()

    def run_full(self):


def recon_traj(enc, net, top, cm):
    n = len(enc)
    n_atoms = top.n_atoms
    x = Variable(torch.from_numpy(enc).type(torch.FloatTensor))
    coords = net.decode(x)
    coords = coords.detach().numpy()
    coords += cm
    coords = coords.reshape((n, n_atoms, 3))
    traj = md.Trajectory(coords, top)
    return traj

def _recon_traj_dir(enc_fn, net, recon_dir, top, cm):
    enc = np.load(enc_fn)
    traj = recon_traj(enc, net, top, cm)

    new_fn = os.path.split(enc_fn)[1]
    base_fn = os.path.splitext(new_fn)[0]
    new_fn = base_fn + ".xtc"
    new_fn = os.path.join(recon_dir, new_fn)
    traj.save(new_fn)

def recon_traj_dir(net, enc_dir, recon_dir, top, cm, n_cores):
    enc_fns = get_fns(enc_dir, "*.npy")
    
    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_recon_traj_dir, net=net, recon_dir=recon_dir, top=top, cm=cm)
    pool.map(f, enc_fns)
    pool.close()

def _calc_labels(enc_fn, net, label_dir):
    enc = np.load(enc_fn)
    try:
        # If a split encoder, only take the latent variables 
        # associated with the classification task
        x = net.encoder1[-1].out_features:
        enc = enc[:,:x]
    except:
        pass
    enc = Variable(torch.from_numpy(enc).type(torch.FloatTensor))
    labels = net.classify(enc)
    labels = labels.detach().numpy()

    new_fn = os.path.split(enc_fn)[1]
    new_fn = os.path.join(label_dir, "lab" + new_fn)
    np.save(new_fn, labels)

def calc_labels(net, enc_dir, label_dir, n_cores):
    enc_fns = get_fns(enc_dir, "*npy")

    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_calc_labels, net=net, label_dir=label_dir,
            net_type='split')
    pool.map(f, enc_fns)
    pool.close()

def get_rmsd_dists(orig_traj, recon_traj):
    n_frames = len(recon_traj)
    if n_frames != len(orig_traj):
        # should raise exception
        print("Can't get rmsds between trajectories of different lengths")
        return
    pairwise_rmsd = []
    for i in range(0, n_frames, 10):
        r = md.rmsd(recon_traj[i], orig_traj[i], parallel=False)[0]
        pairwise_rmsd.append(r)
    pairwise_rmsd = np.array(pairwise_rmsd)
    return pairwise_rmsd

def _rmsd_dists_dir(recon_fn, orig_xtc_dir, ref_pdb):
    recon_traj = md.load(recon_fn, top=ref_pdb.top)
    base_fn = os.path.split(recon_fn)[1]
    orig_fn = os.path.join(orig_xtc_dir, base_fn)
    orig_traj = md.load(orig_fn, top=ref_pdb.top)
    pairwise_rmsd = get_rmsd_dists(orig_traj, recon_traj)
    return pairwise_rmsd

def rmsd_dists_dir(recon_dir, orig_xtc_dir, ref_pdb, n_cores):
    recon_fns = get_fns(recon_dir, "*.xtc")

    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_rmsd_dists_dir, orig_xtc_dir=orig_xtc_dir, ref_pdb=ref_pdb)
    res = pool.map(f, recon_fns)
    pool.close()

    pairwise_rmsd = np.concatenate(res)
    return pairwise_rmsd

def _encode_dir(xtc_fn, net, outdir, top, cm):
    traj = md.load(xtc_fn, top=top)
    n = len(traj)
    n_atoms = traj.top.n_atoms
    x = traj.xyz.reshape((n, 3*n_atoms))-cm
    x = Variable(torch.from_numpy(x).type(torch.FloatTensor))
    if hasattr(net, 'split_inds'):
        lat1, lat2 = net.encode(x)
        output = torch.cat((lat1,lat2),1)
    else:
        output = net.encode(x)
    output = output.detach().numpy()
    new_fn = os.path.split(xtc_fn)[1]
    new_fn = os.path.splitext(new_fn)[0] + ".npy"
    new_fn = os.path.join(outdir, new_fn)
    np.save(new_fn, output)

def encode_dir(net, xtc_dir, outdir, top, n_cores, cm):
    xtc_fns = get_fns(xtc_dir, "*.xtc")

    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_encode_dir, net=net, outdir=outdir, top=top, cm=cm)
    pool.map(f, xtc_fns)
    pool.close()
