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
ROOT.gInterpreter.Declare(
""" 
#include "IBand.cpp"
//#include "GetSphericity.cpp"
#include "TriggerInfo.cpp"
""") # useful cpp functions for histogram vars
### To run code: python main.py -

print('** begin main.py **')

obj = 'tracks_inband' #'tracks_inband' or tracks_all or truthParticle or track or tracks_inband or trackJet
max_per_event = False #maximum number of objects taken per event
num_files = False #number of root files used for sig and bkg
num_epoch = 20
batch_size = 300
set_seed = False
mass = '040' # for lxplus txt files
mult_cut = False
extra_inputs = False
with_disc = False
mult_cuts = {'020':[20,100],'040':[20,100],'060':[40,160],'080':[70,150],'100':[70,290],'120':[100,300]}
norm_avg = False
#print('min tracks = ',min_tracks)

# local = True if running on local Mac, false if on lxplus
# local sets the path for root input files
local = False
dir_path = os.path.dirname(os.path.realpath(__file__))
if 'mariabressan' in dir_path:
    local = True

# Override initial vars if parsed
parser = argparse.ArgumentParser(description='Optional: specify variables')
parser.add_argument('-obj',type=str)
parser.add_argument('-batch_size', type=int, help='batch size. If negative, set number of batches')
parser.add_argument('-num_epoch', type=int)
parser.add_argument('-num_files', type=int)
parser.add_argument('-max_per_event', type=int)
parser.add_argument('-display',type=str, help='display plots? y/Y or n/N. Default to yes if local, no if on lxplus')
parser.add_argument('-set_seed', type = str, help='set seed? string - either n/N or an int')
parser.add_argument('-mass', type=str)
parser.add_argument('-extra_inputs', type=str)
parser.add_argument('-multcut', type=str)
parser.add_argument('-with_disc',type=str)
parser.add_argument('-norm_avg',type=str)

args = parser.parse_args()
if args.num_epoch:
    num_epoch = args.num_epoch
if args.obj:
    obj = args.obj
if args.batch_size:
    batch_size = args.batch_size
if args.num_files:
    num_files = args.num_files
if args.max_per_event:
    max_per_event = args.max_per_event
if args.display:
    if 'y' in args.display or 'Y' in args.display:
        display = True
    elif 'n' in args.display or 'N' in args.display:
        display = False
    else:
        print('display parsed incorrectly: write y or n')
        exit()
else:
    display = local # display defaults to yes if local, no if lxplus
if args.set_seed:
    if 'n' in args.set_seed or 'N' in args.set_seed:
        set_seed = False
    else:
        set_seed = int(args.set_seed)
if args.mass:
    mass = args.mass
if args.multcut:
    if 'y' in args.multcut or 'Y' in args.multcut:
        mult_cut = True
    elif 'n' in args.multcut or 'N' in args.multcut:
        mult_cut = False
    else:
        print('parsed multcut incorrectly. default to '+str(mult_cut))
if args.extra_inputs:
    if 'y' in args.extra_inputs or 'Y' in args.extra_inputs:
        extra_inputs = True
    elif 'n' in args.extra_inputs or 'N' in args.extra_inputs:
        extra_inputs = False
    else:
        print('parsed extra_inputs incorrectly. default to '+str(extra_inputs))
if args.with_disc:
    if 'y' in args.with_disc or 'Y' in args.with_disc:
        with_disc = True
    elif 'n' in args.with_disc or 'N' in args.with_disc:
        with_disc = False
    else:
        print('parsed with_disc incorrectly, default to '+str(with_disc))
if mult_cut:
    min_tracks = mult_cuts[mass][0]
    max_tracks = mult_cuts[mass][1]
else:
    min_tracks = '0'
    max_tracks = '10000'
if args.norm_avg:
    if 'y' in args.norm_avg or 'Y' in args.norm_avg:
        norm_avg = True
    elif 'n' in args.norm_avg or 'N' in args.norm_avg:
        norm_avg = False
    else:
        print('display parsed incorrectly: write y or n')
        exit()
Phi_sizes, F_sizes = (100, 100, 128), (100, 100,100) # specify num of layers and their shapes 
GeV='/1000.' # convert from MeV to GeV

# Set seed
if set_seed:
    os.environ['PYTHONHASHSEED']=str(set_seed)
    np.random.seed(set_seed)
    tf.compat.v1.set_random_seed(set_seed)


# Create str list of root files: signal filenames in sig_fn and background filenames in bkg_fn 
if local:
    sig_fn = ['input/v0.8.3/INST_m100.root'] #for older version: ['input/m40/user.yharris.30303198._000001.INST_STREAM.root','input/m60/user.yharris.30303205._000001.INST_STREAM.root','input/m80/user.yharris.30303211._000001.INST_STREAM.root','input/m100/user.yharris.30303217._000013.INST_STREAM.root','input/m120/user.yharris.30303229._000001.INST_STREAM.root']
    bkg_fn = ['input/v0.8.3/BKG.root'] #for older version: ['input/bkg/user.yharris.29684095._000003.INST_STREAM.root']
else:
    sig_fn = []
    bkg_fn = []
    sig_txt_list = ['/afs/cern.ch/user/m/mbressan/qcdinstantons/Inst_m'+mass+'_v0.8.2-September-05-2022.txt']
    bkg_txt_list = ['/afs/cern.ch/user/m/mbressan/qcdinstantons/Epos_v0.8b-July-21-2022.txt']
    for txt_file in sig_txt_list:
        file = open(txt_file, 'r')
        if not file:
            print('!! cant find txt file: ', txt_file)
        sig_fn = [l.strip('\n\r') for l in file] # reads lines in txt file without \n
    for txt_file in bkg_txt_list:
        file = open(txt_file,'r')
        bkg_fn = [l.strip('\n\r') for l in file]
if num_files:
    sig_fn = sig_fn[0:num_files]
    #bkg_fn = bkg_fn[0:num_files]
    bkg_fn = bkg_fn[0:num_files]

# Chain trees and get RDataframe
sig, bkg = ROOT.TChain("INST_TREE"), ROOT.TChain("INST_TREE")
for fn in sig_fn:
    sig.Add(fn)
for fn in bkg_fn:
    bkg.Add(fn)
rdf = {}
rdf['sig'] = ROOT.RDataFrame(sig)
rdf['bkg'] = ROOT.RDataFrame(bkg)

# Define track variables
for df in rdf:
    sel='(vertex_isPriVtxTrack==1&&track_goodMinbias==1)'
    sel=sel+'||'+'(vertex_isPriVtxTrack==0&&track_nInnermostPixelHits>=0&&track_nPixelHits>=1&&track_nSCTHits>=3&&abs(d0)<10.&&abs(z0)<15.)'
    inst_sel = 'track_nTracks>'+str(min_tracks)+'&&track_nTracks<'+str(max_tracks) 
    rdf[df]=rdf[df].Filter(inst_sel)
    rdf[df]=rdf[df].Filter(inst_sel)
    rdf[df]=rdf[df].Define('track_Bdisc','trackBdisc(track_pt,trackJet_btag_dl1pb,trackJet_to_tracks_map)')
    rdf[df]=rdf[df].Define('Track_z0' ,'track_z0') 
    rdf[df]=rdf[df].Define('Track_d0' ,'track_d0')
    rdf[df]=rdf[df].Define('bs' ,'sqrt(pow(vertex_x[0]-beamPosX,2)+pow(vertex_y[0]-beamPosY,2))')
    rdf[df]=rdf[df].Define('d0' ,'bs-track_d0')
    rdf[df]=rdf[df].Define('z0' ,'beamPosZ-(vertex_z[0]-track_z0)') #z0 with respect to primary vertex always at 0
    rdf[df]=rdf[df].Define('selected', sel)
    band_cut = 15
    arg='ConstructP4Vector_m(track_pt[selected]'+GeV+',track_eta[selected],track_phi[selected],track_m[selected]'+GeV+')'  
    rdf[df]=rdf[df].Define('EtaBand', 'IBandeta('+arg+')')
    rdf[df]=rdf[df].Define('inband','InIBand(track_eta, selected, EtaBand,'+str(band_cut)+'/10.)')
    for cut_type in [['tracks_all','selected'],['tracks_inband','selected&&inband']]:
        rdf[df]=rdf[df].Define(cut_type[0]+'_pt','track_pt['+cut_type[1]+']')
        rdf[df]=rdf[df].Define(cut_type[0]+'_eta','track_eta['+cut_type[1]+']')
        rdf[df]=rdf[df].Define(cut_type[0]+'_phi','track_phi['+cut_type[1]+']')
        rdf[df]=rdf[df].Define(cut_type[0]+'_m','track_m['+cut_type[1]+']')
        rdf[df]=rdf[df].Define(cut_type[0]+'_q','track_q['+cut_type[1]+']')
        rdf[df]=rdf[df].Define(cut_type[0]+'_pixeldEdx','track_pixeldEdx['+cut_type[1]+']')
        rdf[df]=rdf[df].Define(cut_type[0]+'_Bdisc','track_Bdisc['+cut_type[1]+']')

# Convert to numpy array
NN_inputs = [obj+'_pt', obj+'_eta', obj+'_phi']
if extra_inputs:
    if 'track' not in obj:
        print('!!obj must == track to add extra inputs!!')
        exit()
    NN_inputs+=[obj+'_q',obj+'_pixeldEdx']
if with_disc:
    NN_inputs+=[obj+'_Bdisc']
features = NN_inputs + [obj+'_m','track_nTracks', 'theoryWeightNominal','sampleXsec']
np_sig = rdf['sig'].AsNumpy(columns=features)
np_bkg = rdf['bkg'].AsNumpy(columns=features)
sample_weight = np.concatenate((np.full(len(np_sig[obj+'_pt']), 1.),np.full(len(np_bkg[obj+'_pt']), len(np_sig[obj+'_pt'])/len(np_bkg[obj+'_pt']))))
print('test...')
print('Numsigevt: ',len(np_sig[obj+'_pt']))
print('Numbkgevt: ',len(np_bkg[obj+'_pt']))

print('total num samples: ',len(sample_weight), 'array: ',sample_weight)

print('NN_inputs: ',NN_inputs)


# Get weights
# sampleXsec = sample cross section
# len(sampleXsec) = Nevents in sample
# sampleXsec/Nevents in sample = cross section for unit lumi
# theoryWeightNominal = Nevents it should be (theory lumi)
# to get the correct lumi: (cross section for unit limi)(theory lumi)
sig_weight = np_sig['sampleXsec'][0]*np_sig['theoryWeightNominal'][0]
bkg_weight = np_bkg['sampleXsec'][0]*np_bkg['theoryWeightNominal'][0]
print('signal Xsec: ',np_sig['sampleXsec'][0])
print('bkg Xsec: ',np_bkg['sampleXsec'][0])

print('sig_weight: ',sig_weight)
print('bkg_weight: ',bkg_weight)

# loop over events and save nObjects to keep_track
keep_track = [] 
for event in np_sig[obj+'_pt']:
    keep_track.append(len(event))
num_sig_evt = len(keep_track)
print('num sig evt: ',num_sig_evt)
for event in np_bkg[obj+'_pt']:
    keep_track.append(len(event))
num_bkg_evt = len(keep_track)-num_sig_evt
print('num bkg evt: ',num_bkg_evt)

# Set number of train, val, test events based on total number of events 
num_events = len(keep_track)
train, val, test = int(num_events*0.6), int(num_events*0.2), int(num_events*0.2) # suggested percent split
print('train: ',train)
print('val: ',val)
print('test: ',test)

# Initialize shape of X and Y ndarrays
if max_per_event:
    num_tracks = max_per_event
else:
    num_tracks = np.max(keep_track)
X = np.zeros((num_events,num_tracks,len(NN_inputs)),dtype=np.float32) # (number of events, max num of tracks, 3 vector)
Y = np.zeros((num_events,2))
print('** initialized, shape of X: ',np.shape(X),' **')

# Set batch size: If negative, set to number of batches
if batch_size < 0:
    batch_size = int(ceil(-train/batch_size))

print('obj = ',obj)
print('batch_size = ',batch_size)
print('num_epoch = ',num_epoch)
print('num_files = ',num_files)
print('max_per_event = ',max_per_event)
print('display = ',display)
print('set_seeed = ',set_seed)
print('mult cut = ',str(mult_cut), ', tracks from ',min_tracks,'-',max_tracks)
print('norm_avg = ',norm_avg)

# Get pt, eta, phi, m values in X
for i in range(num_sig_evt):
    if max_per_event:
        x = min(max_per_event,keep_track[i])
    else:
        x = keep_track[i]
    for j in range(x):
        for n, input in enumerate(NN_inputs):
            X[i,j,n] = np_sig[input][i][j]
    Y[i,1]=1 # for Y, (0,1) means sig and (1,0) means bkg
for i in range(num_events-num_sig_evt):
    if max_per_event:
        x = min(max_per_event,keep_track[i+num_sig_evt])
    else:
        x = keep_track[i+num_sig_evt]
    for j in range(x):
        for n, input in enumerate(NN_inputs):
            X[i+num_sig_evt,j,n] = np_bkg[input][i][j]
    Y[i+num_sig_evt,0]=1
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

savefile_name = 'm'+mass+'_'+obj+'_'+str(num_epoch)+'epochs_'+str(batch_size)+'batchsize_multcut_'+str(mult_cut)+'_extrainputs_'+str(extra_inputs)+'_'+str(np.random.uniform())[2:]

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
    plt.savefig('ROC_'+str(num_epoch)+'epochs_'+str(batch_size)+'batchsize_'+obj+'_m'+mass+'_tracks'+str(min_tracks)+'-'+str(max_tracks)+'.png')
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