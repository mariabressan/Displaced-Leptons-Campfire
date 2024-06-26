from __future__ import absolute_import, division, print_function
import numpy as np
import energyflow as ef
import math
from energyflow.archs import PFN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical
from sklearn.metrics import roc_auc_score, roc_curve
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
num_epoch = 10
batch_size = 500

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

rdf['sig']=rdf['sig'].Define('wt' ,'pileupWeight*mcEventWeight')
rdf['bkg']=rdf['bkg'].Define('wt' ,'pileupWeight')
rdf['sig']=rdf['sig'].Define('classification' ,'1')
rdf['bkg']=rdf['bkg'].Define('classification' ,'0')

# Convert to numpy array
NN_inputs = ['el_d0', 'el_z0', 'el_dpt']

# add inputs and weights
features = NN_inputs + ['wt','classification']

# convert to numpy
np_sig = rdf['sig'].AsNumpy(columns=features)
np_bkg = rdf['bkg'].AsNumpy(columns=features)

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

for i,input in enumerate(NN_inputs):
    X[:,0,i] = np.concatenate((np_sig[input],np_bkg[input]))

(X_train, X_val, X_test,
 Y_train, Y_val, Y_test) = data_split(X, Y, val=val, test=test)

print('Model summary:')
pfn = PFN(input_dim=3, Phi_sizes=Phi_sizes, F_sizes=F_sizes)

print("train model")
pfn.fit(X_train, Y_train,
        epochs=num_epoch,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        verbose=1)

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
plt.show()