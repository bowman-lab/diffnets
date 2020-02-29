import os
import pickle
import sys
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Trainer:

    def __init__(self,job,myNav):
        """Class to train your DiffNet
        
        Parameters:
        -----------
        job : dictionary with all training parameters
        myNav : Navigator object with directory structure details 
        """
        self.job = job
        self.myNav = myNav
    
    def em_parallel():
        pass

    def apply_exmax():
        pass

    def evaluate_test():
        pass
 
    def train(self):
        job = self.job
        do_em = job['do_em']
        n_epochs = job['n_epochs']
        lr = job['lr'] * lr_fact
        subsample = job['subsample']
        batch_size = job['batch_size']
        batch_output_freq = job['batch_output_freq']
        epoch_output_freq = job['epoch_output_freq']
        test_batch_size = job['test_batch_size']
        em_bounds = job['em_bounds']
        nntype = job['nntype']
        em_batch_size = job['em_batch_size']
        em_n_cores = job['em_n_cores']
 
        n_test = test_inds.shape[0]
        n_batch = np.ceil(train_inds.shape[0]*1.0/subsample/batch_size)

        optimizer = optim.Adam(net.parameters(), lr=lr)
        bce = nn.BCELoss()
        training_loss_full = []
        test_loss_full = []
        epoch_test_loss = []
        best_loss = np.inf
        best_nn = None 
        for epoch in range(n_epochs):
            # shuffle training data
            np.random.shuffle(train_inds)

            # go through mini batches
            running_loss = 0
            i = 0
            for batch_inds in nnutils.chunks(train_inds[::subsample], batch_size):
                minibatch_x = Variable(torch.from_numpy(data[batch_inds]).type(torch.cuda.FloatTensor))
                minibatch_class = Variable(torch.from_numpy(targets[batch_inds]).type(torch.cuda.FloatTensor))

                optimizer.zero_grad()    
                x_pred, latent, class_pred = net(minibatch_x)
                loss = nnutils.my_mse(minibatch_x, x_pred)
                loss += nnutils.my_l1(minibatch_x, x_pred)
                if class_pred is not None:
                    loss += bce(class_pred, minibatch_class)

                #Minimize correlation between latent variables
                n_feat = net.sizes[-1]
                my_c00 = torch.einsum('bi,bo->io', (latent, latent)).mul(1.0/batch_inds.shape[0])
                my_mean = torch.mean(latent, 0)
                my_mean = torch.einsum('i,o->io', (my_mean, my_mean))
                ide = Variable(torch.from_numpy(np.identity(n_feat)).type(torch.cuda.FloatTensor))
                zero_inds = np.where(1-ide.cpu().numpy()>0)
                corr_penalty = nnutils.my_mse(ide[zero_inds], my_c00[zero_inds]-my_mean[zero_inds])
                loss += corr_penalty

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i%batch_output_freq == 0:
                    train_loss = running_loss
                    if i != 0:
                        train_loss /= batch_output_freq
                    training_loss_full.append(train_loss)

                    test_loss = 0
                    for test_batch_inds in nnutils.chunks(test_inds, test_batch_size):
                        test_x = Variable(torch.from_numpy(data[test_batch_inds]).type(torch.cuda.FloatTensor))
                        x_pred, latent, class_pred = net(test_x)
                        loss = self.evaluate_test(test_x,x_pred)
                        #loss = nnutils.my_mse(test_x, x_pred)
                        #print("      ", loss.item(), x_pred.shape, test_x.shape)
                        test_loss += loss.item() * test_batch_inds.shape[0] # mult for averaging across samples, as in train_loss
                    #print("        ", test_loss)
                    test_loss /= n_test # division averages across samples, as in train_loss
                    test_loss_full.append(test_loss)
                    print("    [%s %d, %5d/%d] train loss: %0.6f    test loss: %0.6f" % (label_str, epoch, i, n_batch, train_loss, test_loss))
                    running_loss = 0

                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_nn = copy.deepcopy(net)
                i += 1

            if do_em and hasattr(nntype, "classify"):
                print("    Doing EM")
                targets = self.em_parallel(net, data, train_inds, em_batch_size, indicators, em_bounds, em_n_cores)

            if epoch % epoch_output_freq == 0:
                epoch_test_loss.append(test_loss)
                out_fn = os.path.join(outdir, "epoch_test_loss_%s.npy" % label_str)
                np.save(out_fn, epoch_test_loss)
                out_fn = os.path.join(outdir, "training_loss_%s.npy" % label_str)
                np.save(out_fn, training_loss_full)
                out_fn = os.path.join(outdir, "test_loss_%s.npy" % label_str)
                np.save(out_fn, test_loss_full)
            # nets need be on cpu to load multiple in parallel, e.g. with multiprocessing
                net.cpu()
                out_fn = os.path.join(outdir, "nn_%s_e%d.pkl" % (label_str, epoch))
                pickle.dump(net, open(out_fn, 'wb'))
                net.cuda()
                if do_em and hasattr(nntype, "classify"):
                    out_fn = os.path.join(outdir, "tmp_targets.npy")
                    np.save(out_fn, targets)

            # save best net every epoch
            best_nn.cpu()
            out_fn = os.path.join(outdir, "nn_best_%s.pkl" % label_str)
            pickle.dump(best_nn, open(out_fn, 'wb'))
            best_nn.cuda()
        return best_nn, targets    

    def get_targets(self,act_map,indicators):
        targets = np.zeros((len(indicators), 1))
        targets[:, 0] = act_map[indicators]
        return targets

    def split_test_train(self,frac_test):
        pass
    
    def run(self):
        job = self.job 
        data_dir = self.myNav.whit_data_dir
        outdir = self.myNav.net_dir
        n_latent = job['n_latent']
        layer_sizes = job['layer_sizes']
        n_cores = job['n_cores']
        nntype = job['nntype']
        frac_test = job['frac_test']
        act_map = job['act_map']

        print("  loading data")
        master_fn = os.path.join(data_dir, "master.pdb")
        master = md.load(master_fn)
        out_fn = os.path.join(outdir, "master.pdb")
        master.save(out_fn)
        n_atoms = master.top.n_atoms
        n_features = 3 * n_atoms

        xtc_dir = os.path.join(data_dir, "aligned_xtcs")        

        #Could handle memory better here 
        data = self.myNav.load_traj_coords_dir(xtc_dir, "*.xtc", master.top)
        n_snapshots = len(data)
        print("    size loaded data", data.shape)

        indicator_dir = os.path.join(data_dir, "indicators")
        indicators = nnutils.load_npy_dir(indicator_dir, "*.npy")
        indicators = np.array(indicators, dtype=int)
        targets = self.get_targets(act_map,indicators)

        wm_fn = os.path.join(data_dir, "wm.npy")
        uwm_fn = os.path.join(data_dir, "uwm.npy")
        cm_fn = os.path.join(data_dir, "cm.npy")
        wm = np.load(wm_fn)
        uwm = np.load(uwm_fn)
        cm = np.load(cm_fn).flatten()
        #handle this through split_ae class instead
        #inds = self.get_inds()

        data -= cm

        train_inds, test_inds = self.split_test_train(frac_test)
        n_train = train_inds.shape[0]
        n_test = test_inds.shape[0]
        out_fn = os.path.join(outdir, "train_inds.npy")
        np.save(out_fn, train_inds)
        out_fn = os.path.join(outdir, "test_inds.npy")
        np.save(out_fn, test_inds)
        print("    n train/test", n_train, n_test)

        if hasattr(nntype, 'split_inds'):
            old_net = nntype(layer_sizes[0:2],inds,wm,uwm,master,job.focusDist)
        else:
            old_net = nntype(layer_sizes[0:2],inds,wm,uwm)
        old_net.freeze_weights()

        for cur_layer in range(2,len(layer_sizes)):
            if hasattr(nntype, 'split_inds'):
                old_net = nntype(layer_sizes[0:2],inds,wm,uwm,master,job.focusDist)
            else:
                old_net = nntype(layer_sizes[0:2],inds,wm,uwm)
            net.freeze_weights(old_net)
            net.cuda()
            net, targets = self.train(data, targets, indicators, train_inds, test_inds, net, str(cur_layer), job)
            old_net = net

        #Polishing
        net.unfreeze_weights()
        net.cuda()
        net, targets = self.train(data, targets, indicators, train_inds, test_inds, net, "polish", job, lr_fact=0.1)


class CanonTrainer(Trainer):
    
    def __init__():
        pass

    def evaluate_test(self):

    def get_targets(self,act_map,indicators):
        #Currently a bad implementation because it doesn't use the params
        targ_dir = os.path.join(self.myNav.whit_data_dir, "custom_labels")
        targets = myNav.load_npy_dir(targ_dir)
        return targets

    def split_test_train(self):
        pass

