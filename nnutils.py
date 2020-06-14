# supervised autoencoders
import functools
import glob
import mdtraj as md
import multiprocessing as mp
import numpy as np
import os
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import whiten
import matplotlib.pyplot as plt

from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, roc_curve

class CNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return torch.sigmoid(x)

class split_ae(nn.Module):
    # Move res, focusDist into some other function
    def __init__(self,layer_sizes,inds1,inds2,wm,uwm):
        super(split_ae, self).__init__()
        self.sizes = layer_sizes
        self.n = len(self.sizes)
        self.inds1 = inds1
        self.inds2 = inds2
        self.n_features = len(self.inds1)+len(self.inds2)
        self.wm1 = wm[self.inds1[:,None],self.inds1]
        self.wm2 = wm[self.inds2[:,None],self.inds2]
        self.uwm = uwm
        self.ratio = len(inds1)/(len(inds1)+len(inds2))

        self.encoder1 = nn.ModuleList()
        self.encoder2 = nn.ModuleList()
        self.encoder1.append(nn.Linear(len(inds1),len(inds1)))
        self.encoder2.append(nn.Linear(len(inds2),len(inds2)))
        for i in range(1,self.n-1):
            small_layer_in = int(np.round(self.sizes[i]*self.ratio))
            small_layer_out = int(np.round(self.sizes[i+1]*self.ratio))
            big_layer_in = int(np.round(self.sizes[i] * (1-self.ratio)))
            big_layer_out = int(np.round(self.sizes[i+1] * (1-self.ratio)))
            if small_layer_in < 3:
                small_layer_in = 3
                big_layer_in = self.sizes[i]-3
            if small_layer_out < 3:
                small_layer_out = 3
                big_layer_out = self.sizes[i]-3

            self.encoder1.append(nn.Linear(int(np.round(self.sizes[i]*self.ratio)), int(np.round(self.sizes[i+1]*self.ratio))))
            self.encoder2.append(nn.Linear(int(np.round(self.sizes[i] * (1-self.ratio))), int(np.round(self.sizes[i+1] * (1-self.ratio)))))

        self.decoder = nn.ModuleList()
        for i in range(self.n-1,0,-1):
            self.decoder.append(nn.Linear(self.sizes[i], self.sizes[i-1]))

    @property
    def split_inds(self):
        return True

    def freeze_weights(self,old_net=None):
        vwm = Variable(torch.from_numpy(self.wm1).type(torch.FloatTensor))
        self.encoder1[0].weight.data = vwm
        vz = Variable(torch.from_numpy(np.zeros(len(self.inds1))).type(torch.FloatTensor))
        self.encoder1[0].bias.data = vz
        vwm2 = Variable(torch.from_numpy(self.wm2).type(torch.FloatTensor))
        self.encoder2[0].weight.data = vwm2
        vz2 = Variable(torch.from_numpy(np.zeros(len(self.inds2))).type(torch.FloatTensor))
        self.encoder2[0].bias.data = vz2
        for p in self.encoder1[0].parameters():
            p.requires_grad = False
        for p in self.encoder2[0].parameters():
            p.requires_grad = False
        self.decoder[-1].weight.data = Variable(torch.from_numpy(self.uwm).type(torch.FloatTensor))
        self.decoder[-1].bias.data = Variable(torch.from_numpy(np.zeros(self.n_features)).type(torch.FloatTensor))
        for p in self.decoder[-1].parameters():
            p.requires_grad = False
        
        if old_net:
            n_old = len(old_net.encoder1)
            for i in range(1,n_old):
                self.encoder1[i].weight.data = old_net.encoder1[i].weight.data
                self.encoder1[i].bias.data = old_net.encoder1[i].bias.data
                for p in self.encoder1[i].parameters():
                    p.requires_grad = False
                self.encoder2[i].weight.data = old_net.encoder2[i].weight.data
                self.encoder2[i].bias.data = old_net.encoder2[i].bias.data
                for p in self.encoder2[i].parameters():
                    p.requires_grad = False

    def unfreeze_weights(self):
        n_old = len(self.encoder1)
        for i in range(1,n_old):
           for p in self.encoder1[i].parameters():
               p.requires_grad = True
           for p in self.encoder2[i].parameters():
               p.requires_grad = True

    def encode(self,x):
        x1 = x[:,self.inds1]
        x2 = x[:,self.inds2]
        lat1 = self.encoder1[0](x1)
        lat2 = self.encoder2[0](x2)
        for i in range(1, self.n-1):
            lat1 = F.leaky_relu(self.encoder1[i](lat1))
            lat2 = F.leaky_relu(self.encoder2[i](lat2))
        return lat1, lat2

    def decode(self,latent):
        recon = latent
        for i in range(self.n-2):
            recon = F.leaky_relu(self.decoder[i](recon))
        recon = self.decoder[-1](recon)
        return recon

    def forward(self,x):
        lat1, lat2 = self.encode(x)
        latent = torch.cat((lat1,lat2),1)
        recon = self.decode(latent)

        return recon, latent, None

class split_sae(split_ae):
    def __init__(self, layer_sizes,inds1,inds2,wm,uwm):
        super(split_sae, self).__init__(layer_sizes,inds1,inds2,wm,uwm)

        self.classifier = nn.Linear(self.encoder1[-1].weight.data.shape[0], 1)

    def classify(self, latent):
        return torch.sigmoid(self.classifier(latent))

    def forward(self, x):
        lat1, lat2 = self.encode(x)
        
        label = self.classify(lat1)

        latent = torch.cat((lat1,lat2),1)

        recon = self.decode(latent)
        return recon, latent, label


class ae(nn.Module):
    def __init__(self, layer_sizes,wm=None,uwm=None):
        super(ae, self).__init__()
        self.sizes = layer_sizes
        self.n = len(self.sizes)

        self.encoder = nn.ModuleList()
        for i in range(self.n-1):
            self.encoder.append(nn.Linear(self.sizes[i], self.sizes[i+1]))

        self.decoder = nn.ModuleList()
        for i in range(self.n-1, 0, -1):
            self.decoder.append(nn.Linear(self.sizes[i], self.sizes[i-1]))

    def freeze_weights(self,old_net=None):
        self.encoder[0].weight.data = Variable(torch.from_numpy(self.wm).type(torch.FloatTensor))
        self.encoder[0].bias.data = Variable(torch.from_numpy(np.zeros(len(self.wm))).type(torch.FloatTensor))
        for p in self.encoder[0].parameters():
            p.requires_grad = False
        self.decoder[-1].weight.data = Variable(torch.from_numpy(self.uwm).type(torch.FloatTensor))
        self.decoder[-1].bias.data = Variable(torch.from_numpy(np.zeros(len(self.uwm))).type(torch.FloatTensor))
        for p in self.decoder[-1].parameters():
            p.requires_grad = False

        if old_net:
            n_old = len(old_net.encoder)
            for i in range(1,n_old):
                net.encoder[i].weight.data = old_net.encoder[i].weight.data
                net.encoder[i].bias.data = old_net.encoder[i].bias.data
                for p in net.encoder[i].parameters():
                    p.requires_grad = False

    def unfreeze_weights(self):
        n_old = len(self.encoder)
        for i in range(1,n_old):
           for p in self.encoder[i].parameters():
               p.requires_grad = True

    def encode(self, x):
        # whiten, without applying non-linearity
        latent = self.encoder[0](x)

        # do non-linear layers
        for i in range(1, self.n-1):
            latent = F.leaky_relu(self.encoder[i](latent))

        return latent

    def decode(self, x):
        # do non-linear layers
        recon = x
        for i in range(self.n-2):
            recon = F.leaky_relu(self.decoder[i](recon))
        
        # unwhiten, without applying non-linearity
        recon = self.decoder[-1](recon)

        return recon

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        # None for labels, so number returns same as sae class
        return recon, latent, None


# build classifier based on latent representation from an AE
class classify_ae(nn.Module):
    def __init__(self, n_latent):
        super(classify_ae, self).__init__()
        self.n_latent = n_latent

        self.fc1 = nn.Linear(self.n_latent, 1)

    def classify(self, x):
        return torch.sigmoid(self.fc1(x))

    def forward(self, x):
        return self.classify(x)


class sae(ae):
    def __init__(self, layer_sizes):
        super(sae, self).__init__(layer_sizes)
        
        self.classifier = nn.Linear(self.sizes[-1], 1)

    def classify(self, latent):
        return torch.sigmoid(self.classifier(latent))

    def forward(self, x):
        latent = self.encode(x)
        label = self.classify(latent)
        recon = self.decode(latent)
        return recon, latent, label


class vae(ae):
    def __init__(self, layer_sizes):
        super(vae, self).__init__(layer_sizes)

        # last layer of encoder is mu, also need logvar of equal size
        self.logvar = nn.Linear(self.sizes[-2], self.sizes[-1])

    def encode(self, x):
        # whiten, without applying non-linearity
        latent = self.encoder[0](x)

        # do non-linear layers
        for i in range(1, self.n-2):
            latent = F.leaky_relu(self.encoder[i](latent))

        mu = F.leaky_relu(self.encoder[-1](latent))
        logvar = F.leaky_relu(self.logvar(latent))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        # None for labels, so number returns same as sae class
        return recon, mu, logvar, None


class svae(vae):
    def __init__(self, layer_sizes):
        super(svae, self).__init__(layer_sizes)
        
        self.classifier = nn.Linear(self.sizes[-1], 1)

    def classify(self, latent):
        return torch.sigmoid(self.classifier(latent))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        label = self.classify(z)
        recon = self.decode(z)
        # None for labels, so number returns same as sae class
        return recon, mu, logvar, label


def my_mse(x, x_recon):
    return torch.mean(torch.pow(x-x_recon, 2))


def my_l1(x, x_recon):
    return torch.mean(torch.abs(x-x_recon))


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def get_fns(dir_name, pattern):
    return np.sort(glob.glob(os.path.join(dir_name, pattern)))


def load_npy_dir(dir_name, pattern):
    fns = get_fns(dir_name, pattern)
    all_d = []
    for fn in fns:
        d = np.load(fn)
        all_d.append(d)
    if len(d.shape) == 1:
        all_d = np.hstack(all_d)
    else:
        all_d = np.vstack(all_d)
    return all_d


def load_traj_coords_dir(dir_name, pattern, top):
    fns = get_fns(dir_name, pattern)
    all_d = []
    for fn in fns:
        t = md.load(fn, top=top)
        d = t.xyz.reshape((len(t), 3*top.n_atoms))
        all_d.append(d)
    all_d = np.vstack(all_d)
    return all_d


def split_test_train(n, pct_test=0.2):
    n_test = int(n * pct_test)

    inds = np.arange(n)
    np.random.shuffle(inds)
    train_inds = inds[:-n_test]
    test_inds = inds[-n_test:]

    return train_inds, test_inds


def chunks(arr, chunk_size):
    """Yield successive chunk_size chunks from arr."""
    for i in range(0, len(arr), chunk_size):
        yield arr[i:i + chunk_size]


def split_inds(pdb,resnum,focus_dist):
        res_atoms = pdb.topology.select("resSeq %s" % resnum)
        dist_combos = [res_atoms,np.arange(pdb.n_atoms)]
        dist_combos = np.array(list(itertools.product(*dist_combos)))

        dpdb = md.compute_distances(pdb,dist_combos)
        ind_loc = np.where(dpdb.flatten()<focus_dist)[0]
        inds = np.unique(dist_combos[ind_loc].flatten())

        close_xyz_inds = []
        for i in inds:
            close_xyz_inds.append(i*3)
            close_xyz_inds.append((i*3)+1)
            close_xyz_inds.append((i*3)+2)
        all_inds = np.arange((pdb.n_atoms*3))
        non_close_xyz_inds = np.setdiff1d(all_inds,close_xyz_inds)

        return np.array(close_xyz_inds), non_close_xyz_inds


def _encode_dir(xtc_fn, net_fn, outdir, top, cm):
    net = pickle.load(open(net_fn, 'rb'))
    net.cpu()
    traj = md.load(xtc_fn, top=top)
    n = len(traj)
    n_atoms = traj.top.n_atoms
    x = traj.xyz.reshape((n, 3*n_atoms))-cm
    x = Variable(torch.from_numpy(x).type(torch.FloatTensor))
    if hasattr(net, 'reparameterize'):
        output, _ = net.encode(x)
    else:
        output = net.encode(x)
    output = output.detach().numpy()
    new_fn = os.path.split(xtc_fn)[1]
    new_fn = os.path.splitext(new_fn)[0] + ".npy"
    new_fn = os.path.join(outdir, new_fn)
    np.save(new_fn, output)


def encode_dir(net_fn, xtc_dir, outdir, top, n_cores, cm):
    xtc_fns = get_fns(xtc_dir, "*.xtc")

    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_encode_dir, net_fn=net_fn, outdir=outdir, top=top, cm=cm)
    pool.map(f, xtc_fns)
    pool.close()

def _encode_dir_split(xtc_fn, net_fn, outdir, top, cm):
    net = pickle.load(open(net_fn, 'rb'))
    net.cpu()
    traj = md.load(xtc_fn, top=top)
    n = len(traj)
    n_atoms = traj.top.n_atoms
    x = traj.xyz.reshape((n, 3*n_atoms))-cm
    x = Variable(torch.from_numpy(x).type(torch.FloatTensor))
    if hasattr(net, 'reparameterize'):
        output, _ = net.encode(x)
    else:
        lat1,lat2 = net.encode(x)
    lat1 = lat1.detach().numpy()
    lat2 = lat2.detach().numpy()
    #output = np.concatenate([lat1,lat2],axis=1)
    new_fn = os.path.split(xtc_fn)[1]
    new_fn = os.path.splitext(new_fn)[0] + "lat1.npy"
    new_fn = os.path.join(outdir, new_fn)
    np.save(new_fn, lat1)
    new_fn = os.path.split(xtc_fn)[1]
    new_fn = os.path.splitext(new_fn)[0] + "lat2.npy"
    new_fn = os.path.join(outdir, new_fn)
    np.save(new_fn, lat2)
    output = np.concatenate([lat1,lat2],axis=1)
    new_fn = os.path.split(xtc_fn)[1]
    new_fn = os.path.splitext(new_fn)[0] + ".npy"
    new_fn = os.path.join(outdir, new_fn)
    np.save(new_fn, output)


def encode_dir_split(net_fn, xtc_dir, outdir, top, n_cores, cm):
    xtc_fns = get_fns(xtc_dir, "*.xtc")

    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_encode_dir_split, net_fn=net_fn, outdir=outdir, top=top, cm=cm)
    pool.map(f, xtc_fns)
    pool.close()


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


def _recon_traj_dir(enc_fn, net_fn, recon_dir, top, cm):
    net = pickle.load(open(net_fn, 'rb'))
    net.cpu()
    enc = np.load(enc_fn)
    traj = recon_traj(enc, net, top, cm)

    new_fn = os.path.split(enc_fn)[1]
    base_fn = os.path.splitext(new_fn)[0]
    new_fn = base_fn + ".xtc"
    new_fn = os.path.join(recon_dir, new_fn)
    traj.save(new_fn)


def recon_traj_dir(net_fn, enc_dir, recon_dir, top, cm, n_cores):
    xtc_fns = get_fns(enc_dir, "*npy")

    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_recon_traj_dir, net_fn=net_fn, recon_dir=recon_dir, top=top, cm=cm)
    pool.map(f, xtc_fns)
    pool.close()


def _label_dir(enc_fn, net_fn, label_dir):
    net = pickle.load(open(net_fn, 'rb'))
    net.cpu()
    enc = np.load(enc_fn)
    enc = Variable(torch.from_numpy(enc).type(torch.FloatTensor))
    labels = net.classify(enc)
    labels = labels.detach().numpy()

    new_fn = os.path.split(enc_fn)[1]
    new_fn = os.path.join(label_dir, "lab" + new_fn)
    np.save(new_fn, labels)


def label_dir(net_fn, enc_dir, label_dir, n_cores):
    xtc_fns = get_fns(enc_dir, "*npy")

    pool = mp.Pool(processes=n_cores)
    f = functools.partial(_label_dir, net_fn=net_fn, label_dir=label_dir)
    pool.map(f, xtc_fns)
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

def calc_auc(net_fn,out_fn,data,labels):
    net = pickle.load(open(net_fn, 'rb'))
    net.cpu()
    full_x = torch.from_numpy(data).type(torch.FloatTensor)
    if hasattr(net, "encode"):
        full_x = Variable(full_x.view(-1, 784).float())
        pred_x, latents, pred_class = net(full_x)
        preds = pred_class.detach().numpy()
    else:
        full_x = Variable(full_x.view(-1, 3,32,32).float())
        preds = net(full_x).detach().numpy()
    fpr, tpr, thresh = roc_curve(labels,preds)
    auc = roc_auc_score(labels,preds.flatten())
    print("AUC: %f" % auc)
    #plt.figure()
    #lw = 2
    #plt.plot(fpr, tpr, color='darkorange',
    #     lw=lw, label='ROC curve (area = %f)' % auc)
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    #plt.legend(loc="lower right")
    #plt.savefig(out_fn)
    #plt.close()
    return auc, fpr, tpr
