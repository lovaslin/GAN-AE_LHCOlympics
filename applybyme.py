# This script apply a trained model to a black-box dataset.
# A shaping function is computed to capture the background behaviour at high distance.
# This shaping function can be used to build a reference histogram to use with the BumpHunter algorithm.

import GAN_AE
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import optimize as opt
import pyBumpHunter as BH
import h5py

# Path to dataset
data_path = "data/"

# Name of the GAN-AE model to use
model_name = "RnD_single"

# Load the trained GAN-AE model
print('Loading model')
GAE = GAN_AE.GAN_AE()
GAE.load('models/{}/{}'.format(model_name,model_name))

# Create folder for this model if needed if needed
try:
    os.mkdir('applybyme_results/{}'.format(model_name),0o755)
except:
    pass

# Mode (single/multi, same threshold or not)
mode = 'single'
same = False

dist_all = []

#### FRANKESTEIN ####

# Load the dataset
print('Reading data')
bkg = pd.read_hdf(data_path+'bkgHLF_merged_RnD.h5')
bbi = pd.read_hdf(data_path+'bkgHLF_merged_BB1.h5') 
sig = pd.read_hdf(data_path+'sigHLF_merged_BB1.h5')

# Select the zone to train on (cuts)
# RnD
bkg = bkg[bkg['mjj']>2700]
bkg = bkg[bkg['mjj']<7000]
# BB1
bbi = bbi[bbi['mjj']>2700]
bbi = bbi[bbi['mjj']<7000]
sig = sig[sig['mjj']>2700]
sig = sig[sig['mjj']<7000]

# Here i should split the black box in two but i don't want to

# Remove mjj from data and keep it separately
# It will be used to compute the DisCo term of the loss function
mjj_bkg = bkg['mjj'].values
mjj_bbi = bbi['mjj'].values
mjj_sig = sig['mjj'].values

bkg = bkg.drop(columns='mjj')
bbi = bbi.drop(columns='mjj')
sig = sig.drop(columns='mjj')
bb1 = pd.concat([bbi, sig], axis=1)

# Convert to numpy array and save columns names
var_names = bkg.columns
bkg = bkg.values
bbi = bbi.values
sig = sig.values
bb1 = bb1.values

print('bkg shape = {}'.format(bkg.shape))
print('   bbi.shape={}'.format(bbi.shape))
print('   sig.shape={}'.format(sig.shape))
print('')

# Create folders for this black-box if needed
try:
    os.mkdir('applybyme_results/{}/BB1byme'.format(model_name),0o755)
except:
    pass
try:
    os.mkdir('applybyme_results/{}/BB1byme/shaping_function'.format(model_name),0o755)
    os.mkdir('applybyme_results/{}/BBbyme/BumpHunter'.format(model_name),0o755)
except:
    pass

# Apply the loaded model to RnD dataset and get the distance
    Sbkg,bkg_min,bkg_max = GAE.scale_data(bkg)
    try:
        os.mkdir('applybyme_results/temp',0o755)
    except:
        pass
    if(mode=='single'):
        GAE.apply(Sbkg,bkg_min,bkg_max, var_name=var_names,filename='applybyme_results/temp/RnD',do_latent=False,do_reco=False,do_roc=False,do_auc=False)
        dist_bkg = GAE.distance
    else:
        GAE.multi_apply(Sbkg,bkg_min,bkg_max,var_name=var_names,filename='applybyme_results/temp/RnD',do_latent=False,do_reco=False,do_roc=False,do_auc=False)
        dist_bkg = GAE.distance[:,-1]

# Apply the loaded model to BBi dataset and get the distance
print('Applying GAN-AE')
_,bb1_min,bb1_max = GAE.scale_data(bb1)
Sbbi,_,_ = GAE.scale_data(bbi)
Ssig,_,_ = GAE.scale_data(sig)
print('   Sbbi.shape={}'.format(Sbbi.shape))
print('   Ssig.shape={}'.format(Ssig.shape))
bb1_data = np.empty((Sbbi.shape[0]+Ssig.shape[0],Sbbi.shape[1]))
print('   bb1_data.shape={}'.format(bb1_data.shape))

#BB1
bb1_data[:Sbbi.shape[0]] = Sbbi
bb1_data[Sbbi.shape[0]:] = Ssig
label = np.append(np.zeros(Sbbi.shape[0],dtype=int),np.ones(Ssig.shape[0],dtype=int))
print('   label.shape={}'.format(label.shape))
GAE.apply(bb1_data,bb1_min,bb1_max,var_name=var_names, label=label,filename='applybyme_results/temp/BB1byme', do_latent=False, do_reco=False)
dist_bbi = GAE.distance
print('GAE distance type {}'.format(dist_bbi))
# Save the distance distribution and auc separately
bb1_dist = GAE.distance[0]
sig_dist = GAE.distance[1]
sig_auc=GAE.auc

# Saving the distance on an h5 file
with h5py.File('BB1_distances.h5', "w") as fh5:         
        dset = fh5.create_dataset("bkg", data=bb1_dist)
        dset = fh5.create_dataset("sig", data=sig_dist)
        dset = fh5.create_dataset("aucscore", data=sig_auc)

'''
###############################################################################################################################################################
bb=1
# Append the full distance distribution for bbi to the global dist_all variable
dist_all.append(dist_bbi)

# Do a cut on dist_bkg and compute the shaping function
print('Computing shaping function')
Nth = 50
th_bkg = np.linspace(dist_bkg.min(),dist_bkg.max(),num=100)[Nth]
hist_bkg_cut,bins = np.histogram(mjj_bkg[dist_bkg>th_bkg],bins=40)
hist_bkg,_ = np.histogram(mjj_bkg,bins=bins)

shape_func = hist_bkg_cut/hist_bkg
print('bkg ratio = {}'.format(hist_bkg_cut.sum()/hist_bkg.sum()))

# Fit the shaping function with a polynomial function (order 4)
print('Fitting')
def Fmodel(x,a,b,c,d,e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

x_fit =(bins[1:]+bins[:-1])/2
p,dp = opt.curve_fit(Fmodel,x_fit,shape_func,p0=[1,1,1,0.15,0.2])

for i in range(p.size):
    print('   param {} : {} ({})'.format(i,p[i],dp[i]))
print('')

# Plot the shaping function
F = plt.figure(figsize=(12,8))
plt.title('Shaping function with polynomial fit (order 4)')
plt.hist(bins[:-1],bins=bins,weights=shape_func,histtype='step',linewidth=2,label='hist')
plt.plot(x_fit,Fmodel(x_fit,p[0],p[1],p[2],p[3],p[4]),linewidth=2,label='fit {0:.4f} $x^4$ + {1:.4g} $x^3$ + {2:.4g} $x^2$ + {3:.4g} x + {4:.4g}'.format(p[0],p[1],p[2],p[3],p[4]))
plt.legend(fontsize='large')
plt.xlabel('shaping function',size='large')
plt.savefig('applybyme_results/{}/BB{}/shaping_function/shaping_func.pdf'.format(model_name,bb),bbox_inches='tight')
plt.close(F)

# Define the cut threshold for the black-box
if(same):
    th_bbi = th_bkg
else:
    bestth = 1
    bestrat = 1
    bkgrat = hist_bkg_cut.sum()/hist_bkg.sum()
    th_bbi = np.linspace(dist_bbi.min(),dist_bbi.max(),num=100)
    for th in range(1,th_bbi.size):
        rat = mjj_bbi[dist_bbi>th_bbi[th]].size/mjj_bbi.size
        if(abs(rat/bkgrat-1) < abs(bestrat/bkgrat-1)):
            bestth = th
            bestrat = rat
    th_bbi = th_bbi[bestth]

# Plot the 2 distributions used to produce the shaping function
F = plt.figure(figsize=(12,8))
plt.hist(mjj_bkg[dist_bkg>th_bkg],bins=bins,histtype='step',linewidth=2,color='r',label='bkg with cut')
plt.hist(mjj_bkg,bins=bins,histtype='step',linewidth=2,color='r',linestyle='--',label='bkg without cut')
plt.hist(mjj_bbi[dist_bbi>th_bbi],bins=bins,histtype='step',linewidth=2,color='b',label='BB{} with cut'.format(bb))
plt.hist(mjj_bbi,bins=bins,histtype='step',linewidth=2,color='b',linestyle='--',label='BB{} without cut'.format(bb))
plt.legend(fontsize='large')
plt.yscale('log')
plt.xlabel('mjj',size='large')
plt.savefig('applybyme_results/{}/BB{}/shaping_function/mjj.pdf'.format(model_name,bb),bbox_inches='tight')
plt.close(F)

# Plot the distance distributions
F = plt.figure(figsize=(12,8))
plt.hist(dist_bkg,bins=60,histtype='step',linewidth=2,label='bkg')
plt.hist(dist_bbi,bins=60,histtype='step',linewidth=2,label='bb{}'.format(bb))
plt.legend(fontsize='large')
plt.yscale('log')
plt.xlabel('Euclidean distance')
plt.savefig('applybyme_results/{}/BB{}/shaping_function/distance.pdf'.format(model_name,bb),bbox_inches='tight')
plt.close(F)

# Apply the shaping function to BBi and compare to BBi after cut
hist_bbi,_ = np.histogram(mjj_bbi,bins=bins)
hist_bbi_reshaped = hist_bbi * Fmodel(x_fit,p[0],p[1],p[2],p[3],p[4])
print('bb{} cut ratio = {}'.format(bb,mjj_bbi[dist_bbi>th_bbi].size/mjj_bbi.size))
print('bb{} reshape ratio = {}'.format(bb,hist_bbi_reshaped.sum()/hist_bbi.sum()))

F = plt.figure(figsize=(12,8))
plt.title('BB{} mjj reshaped and cut'.format(bb))
plt.hist(bins[:-1],bins=bins,weights=hist_bbi_reshaped,histtype='step',linewidth=2,label='BB{} reshaped'.format(bb))
hist_bbi_cut,_,_ = plt.hist(mjj_bbi[dist_bbi>th_bbi],bins=bins,histtype='step',linewidth=2,label='BB{} cut'.format(bb))
plt.legend(fontsize='large')
plt.xlabel('mjj')
plt.yscale('log')
plt.savefig('applybyme_results/{}/BB{}/shaping_function/mjj_reshaped.pdf'.format(model_name,bb),bbox_inches='tight')
plt.close(F)
print('th_bkg = {}'.format(th_bkg))
print('th_bb  = {}'.format(th_bbi))


# Apply BumpHunter algorithm to hunt for potential bumps
b = BH.BumpHunter(
    rang=[2700,7000],
    width_min=2,
    width_max=7,
    width_step=1,
    scan_step=1,
    Npe=10000,
    bins=bins,
    Nworker=1,
    seed=666
)

print('')
b.BumpScan(hist_bbi_cut,hist_bbi_reshaped,is_hist=True)

# Print and plot the results
b.PrintBumpInfo()
b.PrintBumpTrue(hist_bbi_cut,hist_bbi_reshaped,is_hist=True)

b.GetTomography(hist_bbi_cut,is_hist=True,filename='applybyme_results/{}/BB{}/BumpHunter/tomography.pdf'.format(model_name,bb))
b.PlotBump(hist_bbi_cut,hist_bbi_reshaped,is_hist=True,filename='applybyme_results/{}/BB{}/BumpHunter/bump.pdf'.format(model_name,bb))
b.PlotBHstat(show_Pval=True,filename='applybyme_results/{}/BB{}/BumpHunter/BH_statistics.pdf'.format(model_name,bb))

print('###########################')
print('')

# Plot all the distance distributions on the same plot
F = plt.figure(figsize=(12,8))
plt.title('Euclidean distance distributions (all black boxes)')
plt.hist(dist_all[0],bins=60,histtype='step',linewidth=2,label='BB1')
plt.legend(fontsize='large')
plt.xlabel('Euclidean distance distribution',size='large')
plt.savefig('applybyme_results/{}/distance_BB1byme.pdf'.format(model_name),bbox_inches='tight')
plt.close(F)
'''