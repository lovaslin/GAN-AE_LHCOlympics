import h5py
# Code to train a GAN_AE instance on a dataset

import GAN_AE
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

print('######################')
print('#######STARTING#######')
print('######################')

# Path to dataset
data_path = "./data/"

model_name = 'RnD_single'
try:
    os.mkdir('train_results/{}'.format(model_name),0o755)
    os.mkdir('models/{}'.format(model_name),0o755)
except:
    pass

# Load the dataset
print('Reading data')
bkg = pd.read_hdf(data_path+'bkgHLF_merged_RnD.h5') 
sig1 = pd.read_hdf(data_path+'sigHLF_merged_RnD.h5')
#sig2 = pd.read_hdf(data_path+'RnD_sig2_HLF.h5')

# Select the zone to train on (cuts)
bkg = bkg[bkg['mjj']>2700]
bkg = bkg[bkg['mjj']<7000]
sig1 = sig1[sig1['mjj']>2700]
sig1 = sig1[sig1['mjj']<7000]
#sig2 = sig2[sig2['mjj']>2700]
#sig2 = sig2[sig2['mjj']<7000]

# Remove mjj from data and keep it separately
# It will be used to compute the DisCo term of the loss function
mjj_bkg = bkg['mjj'].values
mjj_sig1 = sig1['mjj'].values
#mjj_sig2 = sig2['mjj'].values

bkg = bkg.drop(columns='mjj')
sig1 = sig1.drop(columns='mjj')
#sig2 = sig2.drop(columns='mjj')

# Convert to numpy array and save columns names
var_names = bkg.columns
bkg = bkg.values
sig1 = sig1.values
#sig2 = sig2.values

print('   bkg.shape={}'.format(bkg.shape))
print('   sig1.shape={}'.format(sig1.shape))
#print('   sig2.shape={}'.format(sig2.shape))
print('')



# Declare the GAN_AE instance
print('Preparing GAN-AE')
GAE = GAN_AE.GAN_AE(input_dim=var_names.size,
                    hidden_dim = [30,20],
                    latent_dim = 10,
                    dis_dim = [150,100,50],
                    Ncycle = 200,
                    alpha = 0.3,
                    Nmodel = 4,
                    Nselec = 2,
                    pretrain_dis = False
                   )

# Prepare the data
Sbkg,dmin,dmax = GAE.scale_data(bkg)
Ssig1,_,_ = GAE.scale_data(sig1)
#Ssig2,_,_ = GAE.scale_data(sig2)
min_aux = mjj_bkg.min()
max_aux = mjj_bkg.max()
Saux = (mjj_bkg-min_aux)/(max_aux-min_aux)
print('   Sbkg.shape={}'.format(Sbkg.shape))
print('   Ssig1.shape={}'.format(Ssig1.shape))
#print('   Ssig2.shape={}'.format(Ssig2.shape))
print('   Saux.shape={}'.format(Saux.shape))


# Train a single GAN-AE on 100k events
# Validation on 100k events
print('Training GAN-AE')
GAE.train(Sbkg[:100000],Sbkg[100000:200000],[Saux[:100000],Saux[100000:200000]])

# Plot the loss and FoM
print('Doing loss plots')
try:
    os.mkdir('train_results/{}/loss'.format(model_name),0o755)
except:
    pass
GAE.plot(filename='train_results/{}/loss/{}'.format(model_name,model_name))


# Apply the trained GAN-AE to a background/signal1 mixture
# Take 100k backgroud events and all signal1 events
try:
    os.mkdir('train_results/{}/sig1'.format(model_name),0o755)
except:
    pass
print('Applying GAN-AE')
test_data = np.empty((500000+Ssig1.shape[0],Sbkg.shape[1]))
print('   test_data.shape={}'.format(test_data.shape))
test_data[:500000] = Sbkg[200000:700000]
test_data[500000:] = Ssig1
label = np.append(np.zeros(500000,dtype=int),np.ones(Ssig1.shape[0],dtype=int))
print('   label.shape={}'.format(label.shape))
GAE.apply(test_data,dmin,dmax,var_name=var_names,label=label,filename='train_results/{}/sig1/{}'.format(model_name,model_name))

# Save the distance distribution and auc separately
bkg_dist = GAE.distance[0]
sig1_dist = GAE.distance[1]
sig1_auc=GAE.auc

'''
# Apply the trained GAN-AE to a background/signal2 mixture
# Take 100k backgroud events and all signal2 events
try:
    os.mkdir('train_results/{}/sig2'.format(model_name),0o755)
except:
    pass
print('Applying GAN-AE')
test_data = np.empty((500000+Ssig2.shape[0],Sbkg.shape[1]))
print('   test_data.shape={}'.format(test_data.shape))
test_data[:500000] = Sbkg[200000:700000]
test_data[500000:] = Ssig2
label = np.append(np.zeros(500000,dtype=int),np.ones(Ssig2.shape[0],dtype=int))
print('   label.shape={}'.format(label.shape))
GAE.apply(test_data,dmin,dmax,var_name=var_names,label=label,filename='train_results/{}/sig2/{}'.format(model_name,model_name))


# Save the distance distribution and auc separately
sig2_dist = GAE.distance[1]
sig2_auc=GAE.auc
'''

# Saving the distance on an h5 file
with h5py.File('RnD_distances.h5', "w") as fh5:         
        dset = fh5.create_dataset("bkg", data=bkg_dist)
        dset = fh5.create_dataset("sig1", data=sig1_dist)
        #dset = fh5.create_dataset("sig2", data=sig2_dist)

# Plot all distance one single plot
F = plt.figure(figsize=(12,8))
plt.title('Euclidean distance distribution (all)')
plt.hist(bkg_dist,bins=60,histtype='step',linewidth=2,label='background')
plt.hist(sig1_dist,bins=60,histtype='step',linewidth=2,label='signal 1')
#plt.hist(sig2_dist,bins=60,histtype='step',linewidth=2,label='signal 2')
plt.legend(fontsize='large')
plt.xlabel('Euclidean distance',size='large')
plt.savefig('train_results/{}/{}_distance_all.pdf'.format(model_name,model_name),bbox_inches='tight')

# Plot all ROC curve on one single plot
Nbin = 100 # We use 100 points to make the ROC curve
roc_min = min([bkg_dist.min(),sig1_dist.min()])
roc_max = max([bkg_dist.max(),sig1_dist.max()])
step = (roc_max-roc_min)/Nbin
steps = np.arange(roc_min+step,roc_max+step,step)
roc_x = []
roc_x.append(np.array([sig1_dist[sig1_dist>th].size/sig1_dist.size for th in steps]))
#roc_x.append(np.array([sig2_dist[sig2_dist>th].size/sig2_dist.size for th in steps]))
roc_y = np.array([bkg_dist[bkg_dist<th].size/bkg_dist.size for th in steps])
roc_r1 = np.linspace(0,1,100)
roc_r2 = 1-roc_r1

F = plt.figure(figsize=(12,8))
plt.plot(roc_x[0],roc_y,'-',linewidth=2,label='signal 1 auc={0:.4f}'.format(sig1_auc))
#plt.plot(roc_x[1],roc_y,'-',linewidth=2,label='signal 2 auc={0:.4f}'.format(sig2_auc))
plt.plot(roc_r1,roc_r2,'--',label='random class')
plt.legend(fontsize='large')
plt.xlabel('signal efficiency',size='large')
plt.ylabel('background rejection',size='large')
plt.savefig('train_results/{}/{}_ROC_all.pdf'.format(model_name,model_name),bbox_inches='tight')
plt.close(F)


# Save the model
print('Saving GAN-AE')
GAE.save(filename='models/{}/{}'.format(model_name,model_name))



