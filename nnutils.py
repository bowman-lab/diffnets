# supervised autoencoders
import mdtraj as md
import multiprocessing as mp
import numpy as np
import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

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

