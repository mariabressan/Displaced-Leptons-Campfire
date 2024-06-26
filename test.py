import ROOT
import numpy as np
import tensorflow as tf
import energyflow as ef
from datetime import datetime
from energyflow.archs import PFN
from energyflow.utils import data_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from astropy.io import ascii
from astropy.table import Table
import argparse
from math import ceil
import os 

print('** begin main.py **')


sig_fn = ['/eos/atlas/atlascerngroupdisk/phys-susy/displacedleptonsRun3_ANA-SUSY-2022-11/ntuples/v2.1/oneEM/signal_ltrw/unskimmed/SelSelLLP_100_0_10ns.root'] 
bkg_fn = ['/eos/atlas/atlascerngroupdisk/phys-susy/displacedleptonsRun3_ANA-SUSY-2022-11/ntuples/v1p9/oneEM/user.ancsmith.data22_13p6TeV.00437756.physics_Main.deriv.v9conf_VL_23__trees.root/user.ancsmith.00437756.f1305_m2142_p6000.36817393._000004.trees.root'] 

# Chain trees and get RDataframe
sig, bkg = ROOT.TChain("trees_SR_oneEM_noSkim_"), ROOT.TChain("trees_SR_oneEM_")
for fn in sig_fn:
    sig.Add(fn)
for fn in bkg_fn:
    bkg.Add(fn)
rdf = {}
rdf['sig'] = ROOT.RDataFrame(sig)
rdf['bkg'] = ROOT.RDataFrame(bkg)

# Define track variables
for df in rdf:
    # rdf[df]=rdf[df].Filter(cut)
    rdf[df]=rdf[df].Define('el_d0' ,'electron_d0[0]') 
    rdf[df]=rdf[df].Define('el_z0' ,'electron_z0[0]')
    rdf[df]=rdf[df].Define('el_dpt' ,'electron_dpt[0]')

rdf['sig']=rdf['sig'].Define('wt' ,'pileupWeight*mcEventWeight')
rdf['bkg']=rdf['bkg'].Define('wt' ,'pileupWeight')

# Convert to numpy array
NN_inputs = ['el_d0', 'el_z0', 'el_dpt']

# add inputs and weights
features = NN_inputs + ['wt']

# convert to numpy
np_sig = rdf['sig'].AsNumpy(columns=features)
np_bkg = rdf['bkg'].AsNumpy(columns=features)

# define weight
sample_weight = np.concatenate((np_sig['wt'],np_bkg['wt']))

print('test...')
print('Num sig evts: ',len(np_sig['wt']))
print('Num bkg evts: ',len(np_bkg['wt']))

print('total num samples: ',len(sample_weight), 'array: ',sample_weight)

print('NN_inputs: ',NN_inputs)

# Set number of train, val, test events based on total number of events 
num_events = len(sample_weight)
train, val, test = int(num_events*0.6), int(num_events*0.2), int(num_events*0.2) # suggested percent split

print('train: ',train)
print('val: ',val)
print('test: ',test)

# Initialize shape of X and Y ndarrays
X = np.zeros((num_events,len(NN_inputs)),dtype=np.float32) # (number of events, max num of tracks, 3 vector)
Y = np.zeros((num_events,2))
print('** initialized, shape of X: ',np.shape(X),' **')

#print('X: ',X)
#print('Y: ',Y)

print('** put values for X **')

# preprocess by centering jets and normalizing pts

if norm_avg:
    for x in X:
        mask = x[:,0] > 0
        if len(x[mask,0])>0:
            yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()
else:
    for x in X:
        mask = x[:,0] > 0
        max_pt_index = np.argmax(x[:,0])
        x[mask,1] -= x[max_pt_index,1]
        x[mask,2] -= x[max_pt_index,2]
        x[mask,0] /= x[max_pt_index,0] #normalize pt

print('** preprocessed X **')
#print('X: ',X)

# Split the data into training and test samples
(X_train, X_val, X_test, Y_train, Y_val, Y_test, sw_train, sw_val, sw_test) = data_split(X, Y, sample_weight, val=val, test=test)
print('X_train shape:', X_train.shape)
print('Y_train shape: ', Y_train.shape)

'''print('X_train: ',X_train)
print('Y_train: ',Y_train)'''

# build arch
pfn = PFN(input_dim=X_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes)

# train model
pfn.fit(X_train, Y_train,
        epochs=num_epoch,
        batch_size=batch_size,
        validation_data=(X_val, Y_val, sw_val),
        verbose=1,
        #sample_weight=sw_train,
        class_weight={0: num_sig_evt/num_bkg_evt, 1: 1.})

# get predictions on test data
preds = pfn.predict(X_test, batch_size=1000)

# get ROC curve
pfn_fp, pfn_tp, threshs = roc_curve(Y_test[:,1], preds[:,1])

# get area under the ROC curve
auc = roc_auc_score(Y_test[:,1], preds[:,1])
print()
print('PFN AUC:', auc)
print()

savefile_name = 'm'+mass+'_'+'electron_'+str(num_epoch)+'epochs_'+str(batch_size)+'batchsize_multcut_'+str(mult_cut)+'_extrainputs_'+str(extra_inputs)+'_'+str(np.random.uniform())[2:]

# If running local (on mac), make plots. Else (on lxplus), write to .dat file
if local:
    # get multiplicity and mass for comparison
    masses = np.asarray([ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X])
    mults = np.asarray([np.count_nonzero(x[:,0]) for x in X])
    mass_fp, mass_tp, threshs = roc_curve(Y[:,1], masses)
    mult_fp, mult_tp, threshs = roc_curve(Y[:,1], mults)
    #for i in range(len(threshs)):
    #print('mult_fp, mult_tp, thresh: ',mult_fp[i],', ',mult_tp[i],', ',threshs[i])

    # some nicer plot settings 
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True

    # plot the ROC curves
    plt.yscale('log')
    plt.plot(pfn_tp, pfn_fp, '-', color='black', label='PFN')
    plt.plot(mult_tp, mult_fp, '-', color='red', label='Multiplicity')
    #plt.plot(mass_tp, 1-mass_fp, '-', color='blue', label='Jet Mass')

    # axes labels
    plt.xlabel('True Positives')
    plt.ylabel('False Positives')

    # axes limits
    plt.xlim(0, 1)
    plt.ylim(1e-3, 1)

    #make legend and show plot
    plt.legend(loc='lower left', frameon=False)
    plt.savefig('ROC_'+str(num_epoch)+'epochs_'+str(batch_size)+'batchsize_'+'electron_m'+mass+'_tracks'+str(min_tracks)+'-'+str(max_tracks)+'.png')
    if display:
        plt.show()
    plt.clf()

    # get histogram of predictions
    sigmask = Y_test[:,1] > 0
    bkgmask = Y_test[:,0] > 0
    preds_sigmask = preds[sigmask,:]
    preds_bkgmask = preds[bkgmask,:]
    counts_sig, bins_sig = np.histogram(preds_sigmask[:,1], bins=50, weights=np.ones(len(preds_sigmask[:,1]))*sig_weight/len(preds_sigmask[:,1]))
    counts_bkg, bins_bkg = np.histogram(preds_bkgmask[:,1], bins=50, weights=np.ones(len(preds_bkgmask[:,1]))*bkg_weight/len(preds_bkgmask[:,1]))
    counts_sig_tr, bins_sig_tr = np.histogram(keep_track[0:num_sig_evt], bins=50)
    counts_bkg_tr, bins_bkg_tr = np.histogram(keep_track[num_sig_evt:], bins=50)

    plt.stairs(counts_sig, bins_sig, label='signal prediction')
    plt.stairs(counts_bkg, bins_bkg, label='background prediction')
    plt.ylim(max(np.min(counts_sig+counts_bkg)*0.001,100),np.max(counts_sig+counts_bkg)*1.2)
    plt.ylabel('# events')
    plt.xlabel('NN Discriminator')
    plt.yscale('log')
    #plt.stairs(counts_sig_tr, bins_sig_tr/np.max(bins_sig_tr+bins_bkg_tr), label='signal nTracks')
    #plt.stairs(counts_bkg_tr, bins_bkg_tr/np.max(bins_sig_tr+bins_bkg_tr), label='background nTracks')
    plt.legend()
    plt.savefig('./sig_bkg_pred_'+savefile_name+'.png')
    if display:
        plt.show()
    plt.clf()

    plt.stairs(counts_sig/(np.sqrt(counts_bkg)),bins_sig)
    plt.ylabel(r'$S/\sqrt{B}$')
    plt.xlabel('NN Discriminator')
    plt.savefig('./significance_'+savefile_name+'.png')
    if display:
        plt.show()

else:
    mults = np.asarray([np.count_nonzero(x[:,0]) for x in X_test])
    output = Table()
    output['Y'] = Y_test[:,1]
    output['preds'] = preds[:,1]    
    output['mults'] = mults
    ascii.write(output, './output_'+savefile_name+'.dat')
    os.mkdir('NNmodel_'+savefile_name)
    pfn.save('NNmodel_'+savefile_name)
    os.system('tar -czf NNmodel_'+savefile_name+'.tgz NNmodel_'+savefile_name+'/*')
    print('saved NN model')