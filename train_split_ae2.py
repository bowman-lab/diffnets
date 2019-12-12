# include whitening and unwhitending (wuw) as first and last layers
import copy
import exmax
import mdtraj as md
import multiprocessing as mp
import nnutils
import numpy as np
import os
import pickle
import torch.optim as optim
import sys
import torch
import torch.nn as nn
import whiten
import shutil
import glob

from torch.autograd import Variable


TESTING = False


def wrapper(job):
    data_dir = job['data_dir']
    outdir = job['outdir']
    n_latent = job['n_latent']
    layer_sizes = job['layer_sizes']
    n_cores = job['n_cores']
    nntype = job['nntype']
    frac_test = job['frac_test']

    print(os.path.split(outdir)[1])

    print("  loading data")
    master_fn = os.path.join(data_dir, "master.pdb")
    master = md.load(master_fn)
    out_fn = os.path.join(outdir, "master.pdb")
    master.save(out_fn)
    n_atoms = master.top.n_atoms
    n_features = 3 * n_atoms

    xtc_dir = os.path.join(data_dir, "aligned_xtcs")
    #progressive loading
    #label_dir = os.path.join(data_dir, "labels")
    #data = []
    #labels = []
    #for i in range(10):
    #    data_tmp = nnutils.load_traj_coords_dir(xtc_dir, "chunk%s/*.xtc" % i, master.top)
    #    data_tmp = data_tmp[::10]
    #    data.append(data_tmp)
    #    label_tmp = nnutils.load_npy_dir(label_dir, "chunk%s/*.npy" % i)
    #    label_tmp = label_tmp[::10#]
    #    labels.append(label_tmp)
    #data = np.vstack(data)
    #labels = np.hstack(labels)
    ### Done progressive loading
    data = nnutils.load_traj_coords_dir(xtc_dir, "*.xtc", master.top)
    n_snapshots = len(data)
    label_dir = os.path.join(data_dir, "labels")
    labels = nnutils.load_npy_dir(label_dir, "*.npy")
    labels = np.array(labels, dtype=int)
    print("    size loaded data", data.shape, labels.shape)

    if job["canonical"]:
        job["do_em"] = False
        targ_dir = os.path.join(data_dir, "dists")
        print(targ_dir)
        targets = nnutils.load_npy_dir(targ_dir, "tight*.npy")
        targets = targets.flatten()
    else:
        targets = np.zeros((len(labels), 1))
        targets[:, 0] = act_map[labels]

    wm_fn = os.path.join(data_dir, "wm.npy")
    uwm_fn = os.path.join(data_dir, "uwm.npy")
    cm_fn = os.path.join(data_dir, "cm.npy")
    close_xyz_inds_fn = os.path.join(data_dir, "close_xyz_inds.npy")
    inds1 = np.load(close_xyz_inds_fn)
    non_close_xyz_inds_fn = os.path.join(data_dir, "non_close_xyz_inds.npy")
    inds2 = np.load(non_close_xyz_inds_fn)
    wm = np.load(wm_fn)
    #broadcast it to matrix of only atoms within 10A of mutation
    wm1 = wm[inds1[:,None],inds1]
    wm2 = wm[inds2[:,None],inds2]
    uwm = np.load(uwm_fn)
    uwm = uwm
    cm = np.load(cm_fn).flatten()
    # remove cm ahead of time
    data -= cm

    print("  defining train/test sets")
    train_inds, test_inds = nnutils.split_test_train(n_snapshots, pct_test=frac_test)
    if job["Kfold"]:
        train_inds = np.load("blac_stability/DiffNets/noAutoCorrLoss/supervised_comparison/train_inds5.npy")
        test_inds = np.load("blac_stability/DiffNets/noAutoCorrLoss/supervised_comparison/test_inds5.npy")
    if job["balance_load"]:
        train_inds1 = np.where(targets[train_inds].flatten()==1)[0]
        train_inds0 = np.where(targets[train_inds].flatten()==0)[0]
        train_inds0 = np.random.choice(train_inds0,len(train_inds1),replace=False)
        train_inds = np.concatenate([train_inds[train_inds1],train_inds[train_inds0]])
        np.random.shuffle(train_inds)
    n_train = train_inds.shape[0]
    n_test = test_inds.shape[0]
    out_fn = os.path.join(outdir, "train_inds.npy")
    np.save(out_fn, train_inds)
    out_fn = os.path.join(outdir, "test_inds.npy")
    np.save(out_fn, test_inds)
    print("    n train/test", n_train, n_test)

    print("  starting training")
    # set first and last layers to wuw matrices
    old_net = nntype(layer_sizes[0:2],inds1,inds2)
    old_net.encoder1[0].weight.data = Variable(torch.from_numpy(wm1).type(torch.FloatTensor))
    old_net.encoder1[0].bias.data = Variable(torch.from_numpy(np.zeros(len(inds1))).type(torch.FloatTensor))
    old_net.encoder2[0].weight.data = Variable(torch.from_numpy(wm2).type(torch.FloatTensor))
    old_net.encoder2[0].bias.data = Variable(torch.from_numpy(np.zeros(len(inds2))).type(torch.FloatTensor))
    for p in old_net.encoder1[0].parameters():
        p.requires_grad = False
    for p in old_net.encoder2[0].parameters():
        p.requires_grad = False
    old_net.decoder[-1].weight.data = Variable(torch.from_numpy(uwm).type(torch.FloatTensor))
    old_net.decoder[-1].bias.data = Variable(torch.from_numpy(np.zeros(n_features)).type(torch.FloatTensor))
    for p in old_net.decoder[-1].parameters():
        p.requires_grad = False

    # train non-wuw layers
    for cur_layer in range(2, len(layer_sizes)):
        net = nntype(layer_sizes[0:cur_layer+1],inds1,inds2)
        n_new = len(net.encoder1)
        for i in range(n_new):
            print("      ne", net.encoder1[i].weight.data.shape)
            print("      ne", net.encoder2[i].weight.data.shape)
        for i in range(n_new):
            print("      nd", net.decoder[i].weight.data.shape)

        # if not first net, init weights from old_net
        if old_net is not None:
            n_old = len(old_net.encoder1)
            for i in range(n_old):
                net.encoder1[i].weight.data = old_net.encoder1[i].weight.data
                net.encoder1[i].bias.data = old_net.encoder1[i].bias.data
                for p in net.encoder1[i].parameters():
                    p.requires_grad = False
                net.encoder2[i].weight.data = old_net.encoder2[i].weight.data
                net.encoder2[i].bias.data = old_net.encoder2[i].bias.data
                for p in net.encoder2[i].parameters():
                    p.requires_grad = False
            for i in range(-1, -(n_old+1), -1):
                net.decoder[i].weight.data = old_net.decoder[i].weight.data
                net.decoder[i].bias.data = old_net.decoder[i].bias.data
            # don't hold decoder fixed (besides unwhitening)
            #    for p in net.decoder[i].parameters():
            #        p.requires_grad = False
            
            # kepp uw constant
            for p in net.decoder[-1].parameters():
                p.requires_grad = False

        net.cuda()
        net, targets = train(data, targets, labels, train_inds, test_inds, net, str(cur_layer), job)
        old_net = net

    # do poolishing
    if True:
        n_old = len(net.encoder1)
        for i in range(1, n_old):
            for p in net.encoder1[i].parameters():
                p.requires_grad = True
            for p in net.encoder2[i].parameters():
                p.requires_grad = True
            for p in net.decoder[i-1].parameters():
                p.requires_grad = True

        # keep wuw layers contant
        for p in net.encoder1[0].parameters():
            p.requires_grad = False
        for p in net.encoder2[0].parameters():
            p.requires_grad = False
        for p in net.decoder[-1].parameters():
            p.requires_grad = False
        net.cuda()
        net, targets = train(data, targets, labels, train_inds, test_inds, net, "polish", job, lr_fact=0.1)


def train(data, targets, labels, train_inds, test_inds, net, label_str, job, lr_fact=1.0):
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
    lam_cls = 1.0 # equal weight on two mse losses
    lam_corr = 1.0

    n_batch = np.ceil(train_inds.shape[0]*1.0/subsample/batch_size)

    close_xyz_inds_fn = os.path.join(data_dir, "close_xyz_inds.npy")
    inds1 = np.load(close_xyz_inds_fn)
    non_close_xyz_inds_fn = os.path.join(data_dir, "non_close_xyz_inds.npy")
    inds2 = np.load(non_close_xyz_inds_fn)

    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=m)
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
            if hasattr(nntype, 'reparameterize'):
                x_pred, latent, logvar, class_pred = net(minibatch_x)
            else:
                x_pred, latent, class_pred = net(minibatch_x)
            loss = nnutils.my_mse(minibatch_x, x_pred)
            loss += nnutils.my_l1(minibatch_x, x_pred)
            if class_pred is not None:
                loss += bce(class_pred, minibatch_class).mul_(lam_cls)

            # minimize correlation between latent variables, leave auto-correlation free
            if hasattr(nntype, 'reparameterize'):
                corr_penalty = -0.5 * torch.sum(1 + logvar - latent.pow(2) - logvar.exp())
            else:
                n_feat = net.sizes[-1]
                my_c00 = torch.einsum('bi,bo->io', (latent, latent)).mul(1.0/batch_inds.shape[0])
                my_mean = torch.mean(latent, 0)
                my_mean = torch.einsum('i,o->io', (my_mean, my_mean))
                ide = Variable(torch.from_numpy(np.identity(n_feat)).type(torch.cuda.FloatTensor))
                zero_inds = np.where(1-ide.cpu().numpy()>0)
                corr_penalty = nnutils.my_mse(ide[zero_inds], my_c00[zero_inds]-my_mean[zero_inds])
            loss += corr_penalty.mul_(lam_corr)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print("      loss", loss.item())

            if i%batch_output_freq == 0:
                train_loss = running_loss
                if i != 0:
                    train_loss /= batch_output_freq
                training_loss_full.append(train_loss)

                test_loss = 0
                for test_batch_inds in nnutils.chunks(test_inds, test_batch_size):
                    test_x = Variable(torch.from_numpy(data[test_batch_inds]).type(torch.cuda.FloatTensor))
                    if hasattr(nntype, 'reparameterize'):
                        x_pred, latent, logvar, class_pred = net(test_x)
                    else:
                        x_pred, latent, class_pred = net(test_x)
                    loss = nnutils.my_mse(test_x, x_pred)
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

            # for testing
            if TESTING:
                break

        # # do expectation part
        if do_em and hasattr(nntype, "classify"):
            print("    Doing EM")
            targets = em_parallel(net, data, train_inds, em_batch_size, labels, em_bounds, em_n_cores)
            # i = 0
            # for em_batch_inds in nnutils.chunks(train_inds, em_batch_size):
            #     em_batch_x = Variable(torch.from_numpy(data[em_batch_inds]).type(torch.cuda.FloatTensor))
            #     x_pred, latent, class_pred = net(em_batch_x)
            #     cur_labels = class_pred.cpu().detach().numpy()
            #     targets[em_batch_inds] = apply_exmax(cur_labels, labels[em_batch_inds], em_bounds)
            #     if i % 100 == 0:
            #         print("      %d/%d" % (i, n_em))
            #     i += 1

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

    # choose best net instead of last
    # print("  Getting best net")
    # epoch_test_loss = np.array(epoch_test_loss)
    # best_net_ind = epoch_test_loss.argmin() * epoch_output_freq
    # print("    Best net was %d/%d" % (best_net_ind, epoch_output_freq*(len(epoch_test_loss)-1)))
    # net_fn = os.path.join(outdir, "nn_%s_e%d.pkl" % (label_str, best_net_ind))
    # net = pickle.load(open(net_fn, 'rb'))
    # out_fn = os.path.join(outdir, "nn_best_%s.pkl" % label_str)
    # pickle.dump(net, open(out_fn, 'wb'))

    return best_nn, targets


def apply_exmax(inputs):
    cur_labels, labels, em_bounds = inputs
    n_vars = em_bounds.shape[0]

    for i in range(n_vars):
        inds = np.where(labels == i)[0]
        lower = np.int(np.floor(em_bounds[i, 0] * inds.shape[0]))
        upper = np.int(np.ceil(em_bounds[i, 1] * inds.shape[0]))
        cur_labels[inds] = exmax.expectation_range_CUBIC(cur_labels[inds], lower, upper).reshape(cur_labels[inds].shape)

    bad_inds = np.where(np.isnan(cur_labels))
    cur_labels[bad_inds] = 0
    try:
        assert((cur_labels >= 0.).all() and (cur_labels <= 1.).all())
    except AssertionError:
        neg_inds = np.where(cur_labels<0)[0]
        pos_inds = np.where(cur_labels>1)[0]
        bad_inds = neg_inds.tolist() + pos_inds.tolist()
        for iis in bad_inds:
            print("      ", labels[iis], cur_labels[iis])
        print("      #bad neg, pos", len(neg_inds), len(pos_inds))
        #np.save("tmp.npy", tmp_labels)
        cur_labels[neg_inds] = 0.0
        cur_labels[pos_inds] = 1.0
        #sys.exit(1)
    return cur_labels.reshape((cur_labels.shape[0], 1))


def em_parallel(net, data, train_inds, em_batch_size, labels, em_bounds, em_n_cores):
    n_em = np.ceil(train_inds.shape[0]*1.0/em_batch_size)
    freq_output = np.floor(n_em/10.0)
    inputs = []
    i = 0
    for em_batch_inds in nnutils.chunks(train_inds, em_batch_size):
        em_batch_x = Variable(torch.from_numpy(data[em_batch_inds]).type(torch.cuda.FloatTensor))
        if hasattr(net, "decode"):
            if hasattr(net, "reparameterize"):
                x_pred, latent, logvar, class_pred = net(em_batch_x)
            else:
                x_pred, latent, class_pred = net(em_batch_x)
        else:
            class_pred = net(em_batch_x)
        cur_labels = class_pred.cpu().detach().numpy()
        inputs.append([cur_labels, labels[em_batch_inds], em_bounds])
        if i % freq_output == 0:
            print("      %d/%d" % (i, n_em))
        i += 1

    pool = mp.Pool(processes=em_n_cores)
    res = pool.map(apply_exmax, inputs)
    pool.close()

    new_labels = -1 * np.ones((labels.shape[0], 1))
    new_labels[train_inds] = np.concatenate(res)
    return new_labels


def train_classifier(job):
    """For classifying AE"""
    outdir = job['outdir']
    n_epochs = job['n_epochs']
    lr = job['lr']
    n_latent = job['n_latent']
    act_map = job['act_map']
    do_em = job['do_em']
    n_cores = job['n_cores']
    subsample = job['subsample']
    batch_size = job['batch_size']
    batch_output_freq = job['batch_output_freq']
    epoch_output_freq = job['epoch_output_freq']
    test_batch_size = job['test_batch_size']
    em_bounds = job['em_bounds']
    em_fn = job['em_fn']
    em_batch_size = job['em_batch_size']
    em_n_cores = job['em_n_cores']

    if do_em:
        print("  Classifying with em")
    else:
        print("  Classifying without em")

    enc_dir = os.path.join(outdir, "encodings_split")
    encodings = nnutils.load_npy_dir(enc_dir, "*lat1.npy")
    n_snapshots = len(encodings)
    n_latent = encodings.shape[1]

    ### Progressive Loading
    #label_dir = os.path.join(data_dir, "labels")
    #labels = []
    #for i in range(10):
    #    label_tmp = nnutils.load_npy_dir(label_dir, "chunk%s/*.npy" % i)
    #    label_tmp = label_tmp[::10]
    #    labels.append(label_tmp)
    #labels = np.hstack(labels)
    ### End Progressive Loading

    label_dir = os.path.join(data_dir, "labels")
    labels = nnutils.load_npy_dir(label_dir, "*.npy")
    labels = np.array(labels, dtype=int)
    print("    size loaded data", n_snapshots)

    #targets = np.zeros((len(labels), 1))
    #targets[:, 0] = act_map[labels]
    
    targ_dir = os.path.join(data_dir, "dists")
    print(targ_dir)
    targets = nnutils.load_npy_dir(targ_dir, "tight*.npy")
    targets = targets.flatten()

    train_inds = np.load("blac_stability/DiffNets/noAutoCorrLoss/supervised_comparison/train_inds4.npy")
    test_inds = np.load("blac_stability/DiffNets/noAutoCorrLoss/supervised_comparison/test_inds4.npy")

    if do_em:
        out_label_dir = os.path.join(outdir, "%s_labels" % em_fn)
    else:
        out_label_dir = os.path.join(outdir, "labels")
    nnutils.mkdir(out_label_dir)

    #out_fn = os.path.join(outdir, "train_inds.npy")
    #train_inds = np.load(out_fn)
    #out_fn = os.path.join(outdir, "test_inds.npy")
    #test_inds = np.load(out_fn)
    n_train = len(train_inds)
    n_test = len(test_inds)

    n_batch = np.ceil(train_inds.shape[0]*1.0/subsample/batch_size)

    net = nnutils.classify_ae(n_latent)
    net.cuda()
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
            minibatch_x = Variable(torch.from_numpy(encodings[batch_inds]).type(torch.cuda.FloatTensor))
            minibatch_class = Variable(torch.from_numpy(targets[batch_inds]).type(torch.cuda.FloatTensor))

            optimizer.zero_grad()
            class_pred = net(minibatch_x)
            loss = bce(class_pred, minibatch_class)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print("      loss", loss.item())

            if i%batch_output_freq == 0:
                train_loss = running_loss
                if i != 0:
                    train_loss /= batch_output_freq
                training_loss_full.append(train_loss)

                test_loss = 0
                for test_batch_inds in nnutils.chunks(test_inds, test_batch_size):
                    minibatch_class = Variable(torch.from_numpy(targets[test_batch_inds]).type(torch.cuda.FloatTensor))
                    test_x = Variable(torch.from_numpy(encodings[test_batch_inds]).type(torch.cuda.FloatTensor))
                    class_pred = net(test_x)
                    loss = bce(class_pred, minibatch_class)
                    #print("      ", loss.item(), inds.shape[0], n_test)
                    test_loss += loss.item() * test_batch_inds.shape[0] # mult for averaging across samples, as in train_loss
                    #print("        ", test_loss)
                test_loss /= n_test # division averages across samples, as in train_loss
                test_loss_full.append(test_loss)
                print("    [%d, %5d/%d] train loss: %0.6f    test loss: %0.6f" % (epoch, i, n_batch, train_loss, test_loss))
                running_loss = 0

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_nn = copy.deepcopy(net)
            i += 1

        # # do expectation part
        if do_em:
            print("    Doing EM")
            targets = em_parallel(net, encodings, train_inds, em_batch_size, labels, em_bounds, em_n_cores)
            # i = 0
            # for em_batch_inds in nnutils.chunks(train_inds, em_batch_size):
            #     em_batch_x = Variable(torch.from_numpy(encodings[em_batch_inds]).type(torch.cuda.FloatTensor))
            #     class_pred = net(em_batch_x)
            #     cur_labels = class_pred.cpu().detach().numpy()
            #     targets[em_batch_inds] = apply_exmax(cur_labels, labels[em_batch_inds], em_bounds)
            #     if i % 100 == 0:
            #         print("      %d/%d" % (i, n_em))
            #     i += 1

        if epoch % epoch_output_freq == 0:
            epoch_test_loss.append(test_loss)
            out_fn = os.path.join(out_label_dir, "epoch_test_loss.npy")
            np.save(out_fn, epoch_test_loss)
            out_fn = os.path.join(out_label_dir, "training_loss.npy")
            np.save(out_fn, training_loss_full)
            out_fn = os.path.join(out_label_dir, "test_loss.npy")
            np.save(out_fn, test_loss_full)
            # nets need be on cpu to load multiple in parallel, e.g. with multiprocessing
            net.cpu()
            out_fn = os.path.join(out_label_dir, "nn_e%d.pkl" % epoch)
            pickle.dump(net, open(out_fn, 'wb'))
            net.cuda()
            if do_em:
                out_fn = os.path.join(out_label_dir, "tmp_targets.npy")
                np.save(out_fn, targets)

        # save best net every epoch
        best_nn.cpu()
        out_fn = os.path.join(out_label_dir, "nn_best.pkl")
        pickle.dump(best_nn, open(out_fn, 'wb'))
        best_nn.cuda()

    # choose best net instead of last
    # print("  Getting best net")
    # epoch_test_loss = np.array(epoch_test_loss)
    # best_net_ind = epoch_test_loss.argmin() * epoch_output_freq
    # print("    Best net was %d/%d" % (best_net_ind, epoch_output_freq*(len(epoch_test_loss)-1)))
    # net_fn = os.path.join(out_label_dir, "nn_e%d.pkl" % best_net_ind)
    # net = pickle.load(open(net_fn, 'rb'))
    # out_fn = os.path.join(out_label_dir, "nn_best.pkl")
    # pickle.dump(net, open(out_fn, 'wb'))

    enc_dir_lat1 = os.path.join(outdir, "encodings_lat1")
    out_fn = os.path.join(out_label_dir, "nn_best.pkl")
    nnutils.label_dir(out_fn, enc_dir_lat1, out_label_dir, n_cores)


if __name__=='__main__':
    # lr = learning rate, m = momentum 
    data_dir = "blac_stability/wt_t_cabcn"

    # for testing
    if TESTING:
        data_dir = "data/for_testing"

    n_cores = 14
    master_fn = os.path.join(data_dir, "master.pdb")
    master = md.load(master_fn)
    n_atoms = master.top.n_atoms
    n_features = 3 * n_atoms

    n_epochs = 20
    # map from variant index to whether active (1) or inactive (0)
    act_map = np.array([0, 0, 1, 1], dtype=int) #v, wt, t, s
    n_repeats = 1

    jobs = []
    for rep in range(0, n_repeats):
        for lr in [0.0001]: #[0.001, 0.01, 0.1]:
            for n_latent in [25]:
                job = {}
                layer_sizes = [n_features]
                layer_sizes.append(int(n_features))
                layer_sizes.append(int(n_features/4))
                layer_sizes.append(n_latent)
                job['layer_sizes'] = layer_sizes 
                em_bounds = [[0.0, 0.3], #v
                              [0.0, 0.3], # wt
                              [0.60, 0.90], #t
                              [0.60, 0.90]]
                job['do_em'] = False
                job['em_bounds'] = np.array(em_bounds)
                job['em_fn'] = 'em1'
                job['em_batch_size'] = 150
                job['em_n_cores'] = 14
                job['nntype'] = nnutils.split_sae
                job['lr'] = lr
                job['n_latent'] = n_latent
                job['rep'] = rep
                job['n_epochs'] = n_epochs
                if TESTING:
                    job['n_epochs'] = 3
                job['data_dir'] = data_dir
                job['act_map'] = act_map
                job['n_cores'] = n_cores
                job['batch_size'] = 32
                job['batch_output_freq'] = 500
                job['epoch_output_freq'] = 2
                job['test_batch_size'] = 1000
                job['frac_test'] = 0.1
                job['subsample'] = 10
                job['canonical'] = True
                job['balance_load'] = False
                job['Kfold'] = True
                jobs.append(job)

                #continue

                job2 = dict(job)
                job2['nntype'] = nnutils.split_ae
                jobs.append(job2)

                continue
                 
                job2 = dict(job)
                job2['nntype'] = nnutils.split_sae
                job2['do_em'] = True
                jobs.append(job2)

                continue

                job2 = dict(job)
                job2['nntype'] = nnutils.sae
                jobs.append(job2)

                job2 = dict(job)
                job2['nntype'] = nnutils.ae
                jobs.append(job2)

                job2 = dict(job)
                job2['nntype'] = nnutils.sae
                job2['do_em'] = False
                jobs.append(job2)



    print(len(jobs))
   #if True:
    for job in jobs:
        print("starting")
        outdir = "blac_stability/DiffNets/AutoCorrLoss/DiffNets_classify/%s_e%d_lr%1.6f_lat%d_3-22_r%d_k5" % (job['nntype'].__name__, job['n_epochs'], job['lr'], job['n_latent'], job['rep'])
        if job['do_em'] and hasattr(job['nntype'], 'classify'):
            outdir = outdir + "_" + job['em_fn']
        if TESTING:
            outdir = "test_" + outdir
        outdir = os.path.abspath(outdir)
        job['outdir'] = outdir

        if os.path.exists(outdir):
            out_label_dir = os.path.join(outdir, "%s_labels" % job['em_fn'])
            if job['nntype'] == nnutils.ae and not os.path.exists(out_label_dir):
                job2 = dict(job)
                job2['do_em'] = True
                train_classifier(job2)
                continue
            else:
                print("skipping %s because it already exists" % outdir)
                continue
        cmd = "mkdir %s" % outdir
        os.system(cmd)
        #os.system("cp %s %s/" % (sys.argv[0], outdir))

        wrapper(job)
        #outdir="blac_stability/split_ae/not_finished/split_sae_e20_lr0.000100_lat30_r0"
        net_fn = os.path.join(outdir, "nn_best_polish.pkl")

        # don't let everything try to use all cores
        torch.set_num_threads(1)

        top_fn = os.path.join(data_dir, "master.pdb")
        ref_pdb = md.load(top_fn)
        #ref_pdb_slice = ref_pdb.atom_slice(close_inds)
        n_atoms = ref_pdb.n_atoms
        n_features = 3 * n_atoms
        wm_fn = os.path.join(data_dir, "wm.npy")
        uwm_fn = os.path.join(data_dir, "uwm.npy")
        cm_fn = os.path.join(data_dir, "cm.npy")
        wm = np.load(wm_fn)
        uwm = np.load(uwm_fn)
        cm = np.load(cm_fn).flatten()
        n_cores = 14
        # save encoding for all data
        print("  saving encoding of all data")
        enc_dir = os.path.join(outdir, "encodings")
        nnutils.mkdir(enc_dir)
        orig_xtc_dir = os.path.join(data_dir, "aligned_xtcs")
        
        #nnutils.encode_dir(net_fn, orig_xtc_dir, enc_dir, ref_pdb.top, n_cores, cm)
        nnutils.encode_dir_split(net_fn, orig_xtc_dir, enc_dir, ref_pdb.top, n_cores, cm)
        enc_dir_split = os.path.join(outdir, "encodings_split")
        nnutils.mkdir(enc_dir_split)
        enc_dir_lat1 = os.path.join(outdir, "encodings_lat1")
        nnutils.mkdir(enc_dir_lat1)
        for fn in glob.glob(os.path.join(enc_dir,"*lat*.npy")):
            shutil.move(fn, "%s/." % enc_dir_split)
        for fn in glob.glob(os.path.join(enc_dir_split,"*lat1.npy")):
            shutil.copy(fn, "%s/." % enc_dir_lat1)
            

        # get reconstructed traj
        print("  reconstructing trajectory")
        recon_dir = os.path.join(outdir, "recon_trajs")
        nnutils.mkdir(recon_dir)
        nnutils.recon_traj_dir(net_fn, enc_dir, recon_dir, ref_pdb.top, cm, n_cores)
        #if True:
        if hasattr(job['nntype'], "classify"):
            print("  Saving labels")
            label_dir = os.path.join(outdir, "labels")
            nnutils.mkdir(label_dir)
            nnutils.label_dir(net_fn, enc_dir_lat1, label_dir, n_cores)
        else:
            print("  Training classifier on latent space")
            job2 = dict(job)
            job2['do_em'] = False
            train_classifier(job2)
            #job2 = dict(job)
            #job2['do_em'] = True
            #train_classifier(job2)

        # get distribution of rmsds
        print("  getting rmsds")
        n_cores = 1 # compensate for parallel=False not working in mdtraj
        os.environ["OMP_NUM_THREADS"] = "1"
        orig_xtc_dir = os.path.join(data_dir, "aligned_xtcs")
        pairwise_rmsd = nnutils.rmsd_dists_dir(recon_dir, orig_xtc_dir, ref_pdb, n_cores)
        rmsd_fn = os.path.join(outdir, "rmsd.npy")
        np.save(rmsd_fn, pairwise_rmsd)

