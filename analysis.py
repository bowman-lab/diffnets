import mdtraj as md
import nnutils

class NetExplorer:

    def __init__(self, net, nav):
        self.net = net
        self.nav = nav

    def encode_data(self):
        enc_dir = os.path.join(self.nav.net_dir, "encodings")
        nnutils.mkdir(enc_dir)
        #Handle normal AE or split AE here

    def recon_traj(self):
        recon_dir = os.path.join(self.nav.net_dir, "recon_trajs")
        nnutils.mkdir(recon_dir)

    def get_labels(self):
        label_dir = os.path.join(self.nav.net_dir, "labels")
        nnutils.mkdir(label_dir)

    def get_rmsd(self):
        rmsd_fn = os.path.join(self.nav.net_dir, "rmsd.npy")
        rmsd = 

    def morph()

    def project_labels()

    def check_loss()

    def clust_enc()

    def find_feats()

    def run(self):
        self.encode_data()
        self.recon_traj()
        self.get_labels()
        self.get_rmsd()
