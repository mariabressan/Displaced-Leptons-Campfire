from __future__ import absolute_import, division, print_function
import numpy as np
import energyflow as ef
import math
from energyflow.archs import PFN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import ROOT
import numpy as np
import tensorflow as tf
from astropy.io import ascii
from astropy.table import Table
import argparse
from math import ceil
import os 

# network architecture parameters
Phi_sizes, F_sizes = (20, 20), (20, 20)

# network training parameters
num_epoch = 2
batch_size = 100

savefile_name = 'test1'
################################################################################

print("load data")
sig_fn = ['/eos/atlas/atlascerngroupdisk/phys-susy/displacedleptonsRun3_ANA-SUSY-2022-11/ntuples/v2.1/oneEM/signal_ltrw/unskimmed/SelSelLLP_100_0_10ns.root'] 
bkg_fn = ['/eos/atlas/atlascerngroupdisk/phys-susy/displacedleptonsRun3_ANA-SUSY-2022-11/ntuples/v1p9/oneEM/user.ancsmith.data22_13p6TeV.00437756.physics_Main.deriv.v9conf_VL_23__trees.root/user.ancsmith.00437756.f1305_m2142_p6000.36817393._000004.trees.root'] 
sig, bkg = ROOT.TChain("trees_SR_oneEM_noSkim_"), ROOT.TChain("trees_SR_oneEM_")
for fn in sig_fn:
    sig.Add(fn)
for fn in bkg_fn:
    bkg.Add(fn)
rdf = {}
rdf['sig'] = ROOT.RDataFrame(sig)
rdf['bkg'] = ROOT.RDataFrame(bkg)

for df in rdf:
    # rdf[df]=rdf[df].Filter(cut)
    rdf[df]=rdf[df].Define('el_d0' ,'electron_d0[0]') 
    rdf[df]=rdf[df].Define('el_z0' ,'electron_z0[0]')
    rdf[df]=rdf[df].Define('el_dpt' ,'electron_dpt[0]')
    rdf[df]=rdf[df].Define('el_f3' ,'electron_f3[0]')
    rdf[df]=rdf[df].Define('el_f1' ,'electron_f1[0]')
    rdf[df]=rdf[df].Define('ln_el_d0' ,'log(electron_d0[0])') 
    rdf[df]=rdf[df].Define('ln_el_z0' ,'log(electron_z0[0])')

rdf['sig']=rdf['sig'].Define('wt' ,'pileupWeight*mcEventWeight')
rdf['bkg']=rdf['bkg'].Define('wt' ,'pileupWeight')
rdf['sig']=rdf['sig'].Define('classification' ,'1')
rdf['bkg']=rdf['bkg'].Define('classification' ,'0')

# Convert to numpy array
NN_inputs = ['el_d0', 'el_z0', 'el_dpt','el_f1','el_f3']

# add inputs and weights
features = NN_inputs + ['wt','classification']

# convert to numpy
np_sig = rdf['sig'].AsNumpy(columns=features)
np_bkg = rdf['bkg'].AsNumpy(columns=features)

for var in np_sig:
    print(var)
    counts_test, bins_test = np.histogram(np.concatenate((np_sig[var],np_bkg[var])), bins=50)
    counts_sigvar, bins_sigvar = np.histogram(np_sig[var], bins=bins_test, weights=np_sig['wt']/np.sum(np_sig['wt']))
    counts_bkgvar, bins_bkgvar = np.histogram(np_bkg[var], bins=bins_test, weights=np_bkg['wt']/np.sum(np_bkg['wt']))
    plt.yscale('log')
    plt.stairs(counts_sigvar, bins_test, label='sig', color="blue")
    plt.stairs(counts_bkgvar, bins_test, label='bkg', color="red")
    plt.legend()
    plt.title(f'{var}')
    plt.savefig(var+'.png')
    plt.clf()
    print("done")


'''# transform vars for input
scaler = StandardScaler()
for key in np_sig:
    np_sig[key] -= np.average(np_sig[key])
    np_sig[key] /= np.sum(np_sig[key])
    np_bkg[key] -= np.average(np_bkg[key])
    np_bkg[key] /= np.sum(np_bkg[key])
'''
num_sig_evt = len(np_sig['classification'])
num_bkg_evt = len(np_bkg['classification'])
print("num sig evt: ",num_sig_evt)
print("num bkg evt: ",num_bkg_evt)

# define weight
sample_weight = np.concatenate((np_sig['wt'],np_bkg['wt']))
num_events = len(sample_weight)
X = np.zeros((num_events,1, len(NN_inputs)))
Y = to_categorical(np.concatenate((np_sig['classification'],np_bkg['classification'])), num_classes=2)
train, val, test = int(num_events*0.6), int(num_events*0.2), int(num_events*0.2) # suggested percent split
print("train, val, test = ",train, ", ",val,", ",test)

for i,input in enumerate(NN_inputs):
    X[:,0,i] = np.concatenate((np_sig[input],np_bkg[input]))

(X_train, X_val, X_test,
 Y_train, Y_val, Y_test,
 sw_train, sw_val, sw_test) = data_split(X, Y, sample_weight, val=val, test=test)

print('Model summary:')
pfn = PFN(input_dim=len(NN_inputs), Phi_sizes=Phi_sizes, F_sizes=F_sizes)

print("train model")
pfn.fit(X_train, Y_train,
        epochs=num_epoch,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        verbose=1,
        #sample_weight=sw_train,
        class_weight={0: num_sig_evt/num_bkg_evt, 1: 1.})

print("get predictions on test data")
preds = pfn.predict(X_test, batch_size=1000)

# get ROC curve
pfn_fp, pfn_tp, threshs = roc_curve(Y_test[:,1], preds[:,1])

# get area under the ROC curve
auc = roc_auc_score(Y_test[:,1], preds[:,1])
print()
print('PFN AUC:', auc)
print()

# some nicer plot settings 
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True

# plot the ROC curves
plt.plot(pfn_tp, 1-pfn_fp, '-', color='black', label='PFN')

# axes labels
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')

# axes limits
plt.xlim(0, 1)
plt.ylim(0, 1)

# make legend and show plot
plt.legend(loc='lower left', frameon=False)
plt.savefig('./ROC_'+savefile_name+'.png')
plt.clf()
'''
# get histogram of predictions
sigmask = Y_test[:,1] > 0
bkgmask = Y_test[:,0] > 0
preds_sigmask = preds[sigmask,:]
preds_bkgmask = preds[bkgmask,:]
counts_sig, bins_sig = np.histogram(preds_sigmask[:,1], bins=50, weights=np.ones(len(preds_sigmask[:,1]))*sw_test[sigmask]/len(preds_sigmask[:,1]))
counts_bkg, bins_bkg = np.histogram(preds_bkgmask[:,1], bins=50, weights=np.ones(len(preds_bkgmask[:,1]))*sw_test[bkgmask]/len(preds_bkgmask[:,1]))
print("preds sig ", preds_sigmask[:,1])
print("preds bkg ", preds_bkgmask[:,1])
#counts_sig_tr, bins_sig_tr = np.histogram(keep_track[0:num_sig_evt], bins=50)
#counts_bkg_tr, bins_bkg_tr = np.histogram(keep_track[num_sig_evt:], bins=50)

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
plt.clf()

plt.stairs(counts_sig/(np.sqrt(counts_bkg)),bins_sig)
plt.ylabel(r'$S/\sqrt{B}$')
plt.xlabel('NN Discriminator')
plt.savefig('./significance_'+savefile_name+'.png')'''