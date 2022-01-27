# Implement a class that build and train a GAN_AE achitecture

import numpy as np
from functools import partial
import concurrent.futures as thd
from threading import RLock
lock = RLock()


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as Kl
from tensorflow.keras import models as Km
from tensorflow.keras.losses import binary_crossentropy as bc
from tensorflow.keras import optimizers as Kopt
from tensorflow.keras.initializers import glorot_normal as gn

from sklearn.metrics import auc

import matplotlib.pyplot as plt

import h5py
from random import randrange

# Define loss function for AE only model
def MeanL2Norm(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true),axis=1)))


# Define DisCo term
def DisCo(y_true, y_pred,var_1,var_2,power=1,alph=1.0):
    '''
    Taken from https://github.com/gkasieczka/DisCo/blob/master/Disco_tf.py
    I just removed the 'normedweight' thing since I don't need it here.
    '''
    
    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])
    
    yy = tf.transpose(xx)
    
    amat = tf.math.abs(xx-yy)
    
    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
    
    yy = tf.transpose(xx)
    bmat = tf.math.abs(xx-yy)
    
    amatavg = tf.reduce_mean(amat, axis=1)
    bmatavg = tf.reduce_mean(bmat, axis=1)
    
    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)
    Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg)
    
    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)
    Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg)
    
    ABavg = tf.reduce_mean(Amat*Bmat,axis=1)
    AAavg = tf.reduce_mean(Amat*Amat,axis=1)
    BBavg = tf.reduce_mean(Bmat*Bmat,axis=1)
    
    if power==1:
        dCorr = tf.reduce_mean(ABavg)/tf.math.sqrt(tf.reduce_mean(AAavg)*tf.reduce_mean(BBavg))
    elif power==2:
        dCorr = (tf.reduce_mean(ABavg))**2/(tf.reduce_mean(AAavg)*tf.reduce_mean(BBavg))
    else:
        dCorr = (tf.reduce_mean(ABavg)/tf.math.sqrt(tf.reduce_mean(AAavg)*tf.reduce_mean(BBavg)))**power
    
    return alph*dCorr


# Define the full loss of the AE+D model
def _full_loss(y_true, y_pred,in_true,in_reco,eps):
    return bc(y_true, y_pred)+eps*MeanL2Norm(in_true,in_reco)

# Define the full loss of the AE+D model (with extra DisCo term)
#def _full_loss_DisCo(y_true, y_pred,in_true,in_reco,eps,alph,var_1,power): 
#    var_2 = K.sqrt(K.sum(K.square(y_pred - y_true),axis=1))
#    DC = DisCo(var_1,var_2,power)
#    return bc(y_true, y_pred)+eps*MeanL2Norm(in_true,in_reco)+alph*DC


class GAN_AE():
    '''
    Provides all the methods to build, train and test one or several GAN-AE models.
    
    Hyperarameters :
        input_dim : Dimention of the AE input
        hidden_dim : Dimension of hdden layers between input and latent space (AE architecture)
        latent_dim : Dimention of AE latent space (bottleneck size)
        dis_dim : dimension of the discriminator hiden layers (D architecture)
        epsilon : Combined loss function parameter
        alpha : Disco term parameter
        power : The power used when calculating DisCo term
        NGAN : Number of GAN (AE+discriminant) epochs per training cycle
        ND : number of discriminant only epochs per training cycle
        Ncycle : Total numbur of training cycle
        batch_size : The batchsize used for training 
        early_stop : specify the number of epochs without improvment that trigger the early stpping (if None, no early stopping)
        pretrain_AE : specify if the AE should be pretrained separatly before the GAN training
        pretrain_dis : specify if the discriminant should be pretrained separatly before the GAN training
        Nmodel : Total number of trained GAN-AE model
        Nselec : Total number of selected GAN-AE model for averaging
        Nworker : Maximum number of therads to un in parallel (must be 1 for tensorflow version>1.14.0)
    '''
    
    def __init__(self,
                 input_dim=10,hidden_dim=[7],latent_dim=5,
                 dis_dim=[100,70,50],
                 epsilon=0.2,alpha=None,power=1,
                 NGAN=4,ND=10,batch_size=1024,Ncycle=60,early_stop=5,pretrain_AE=False,pretrain_dis=True,
                 Nmodel=60,Nselec=10,Nworker=4
                ):
        
        # Initialize parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dis_dim = dis_dim
        self.epsilon = epsilon
        self.alpha = alpha
        self.power = power
        self.NGAN = NGAN
        self.ND = ND
        self.batch_size = batch_size
        self.Ncycle = Ncycle
        self.early_stop = early_stop
        self.pretrain_AE = pretrain_AE
        self.pretrain_dis = pretrain_dis
        self.Nmodel = Nmodel
        self.Nselec = Nselec
        self.Nworker = Nworker
        
        # Initialize results
        self.FoM = []
        self.AE_weights = []
        self.dis_weights = []
        self.loss = []
        self.distance = []
        self.auc = []
        
        return
    
    # Prepare the data by rescaling everything between 0 and 1 (min-max scaler)
    def scale_data(self,data):
        '''
        MinMax scaler (rescale the data between 0 and 1).
        
        Argument :
            data : The data to be scaled (given as a numpy array).
        
        Returns :
            res : The rescaled data.
            dmin : The original minimum of all variables (for inverting the scaling).
            dmax : The original maximum of all variables (for inverting the scaling).
        '''
        
        # Get the min and max of all variables
        dmin = np.array([data[:,i].min() for i in range(data.shape[1])])
        dmax = np.array([data[:,i].max() for i in range(data.shape[1])])
        
        # Apply the min-max scaling
        res = np.zeros(shape=data.shape)
        for i in range(data.shape[1]):
            res[:,i] = (data[:,i]-dmin[i])/(dmax[i]-dmin[i])
        
        return res,dmin,dmax
    
    # Invert the min-max scaler
    def restore_data(self,data,dmin,dmax):
        '''
        Invert the MinMax scaler to restore the original data scale.
        
        Argument :
            data : The scaled data to be restored (given as a numpy array).
            dmin : The original minimum of all variables.
            dmax : The original maximum of all variables.
        
        Return :
            res : The restored data.
        '''
        
        # Revert the min-max scaling
        res = np.zeros(shape=data.shape)
        for i in range(data.shape[1]):
            res[:,i]= (data[:,i]*(dmax[i]-dmin[i]))+dmin[i]
        
        return res
    
    # Build the GAN-AE architecture
    def build(self,display=True):
        '''
        Method that builds a GAN-AE architecture and return its compiled components.
        
        Arguments :
            display : Specify if the detail of the built model shoud pe printed after building it.
                      Default to True.
        
        Returns :
            En : The compiled encoder keras model.
            
            AE : The compiled AE keras model (encoder+decoder).
            
            D : The compiled discriminator keras model (MLP).
            
            GAN : The full compiled GAN-AE keras model (AE+D).
        '''
        
        # Encoder input
        En_in = Kl.Input(shape=(self.input_dim,),name='En_input')
        
        # Encoder hidden layers
        for i in range(len(self.hidden_dim)):
            if i==0:
                En_h = Kl.Dense(self.hidden_dim[i],activation='linear',kernel_initializer=gn(),name='encode{}'.format(i+1))(En_in)
            else:
                En_h = Kl.Dense(self.hidden_dim[i],activation='linear',kernel_initializer=gn(),name='encode{}'.format(i+1))(En_h)
            En_h = Kl.Dropout(0.2)(En_h)
            En_h = Kl.ReLU()(En_h)
        
        # Encoder output
        En_co = Kl.Dense(self.latent_dim,activation='relu',kernel_initializer=gn(),name='En_code')(En_h)
        En = Km.Model(En_in,En_co,name='Encoder')
        
        # Decoder
        De_in = Kl.Input(shape=(self.latent_dim,),name='De_input')
        N = len(self.hidden_dim)
        for i in range(N):
            if i==0:
                De_h = Kl.Dense(self.hidden_dim[N-i-1],activation='linear',kernel_initializer=gn(),name='encode{}'.format(i+1))(De_in)
            else:
                De_h = Kl.Dense(self.hidden_dim[N-i-1],activation='linear',kernel_initializer=gn(),name='encode{}'.format(i+1))(De_h)
            De_h = Kl.Dropout(0.2)(De_h)
            De_h = Kl.ReLU()(De_h)
        De_out = Kl.Dense(self.input_dim,activation='sigmoid',kernel_initializer=gn(),name='De_outout')(De_h)
        De = Km.Model(De_in,De_out,name='Decoder')
        
        # Full generator/AE
        AE_in = Kl.Input(shape=(self.input_dim,),name='Gen_input')
        AE_mid = En(AE_in)
        AE_out = De(AE_mid)
        AE = Km.Model(AE_in,AE_out,name='Generator (AE)')
        AE.compile(loss=MeanL2Norm,optimizer=Kopt.Adam(lr=0.0002,beta_1=0.5))
        
        # Discriminator
        Din  = Kl.Input(shape=(self.input_dim,))
        for i in range(len(self.dis_dim)):
            if i==0:
                Dh  = Kl.Dense(self.dis_dim[i],activation='linear',kernel_initializer=gn(),name='Dis{}'.format(i+1))(Din)
            else:
                Dh  = Kl.Dense(self.dis_dim[i],activation='linear',kernel_initializer=gn(),name='Dis{}'.format(i+1))(Dh)
            Dh  = Kl.Dropout(0.2)(Dh)
            Dh = Kl.LeakyReLU(alpha=0.2)(Dh)
        Dout = Kl.Dense(1,activation='sigmoid',name='Dis_output')(Dh)
        D = Km.Model(Din,Dout,name='Discriminant')
        D.compile(loss=bc,optimizer=Kopt.Adam(lr=0.0002,beta_1=0.5))
        
        
        # Full GAN-AE
        GANin = Kl.Input(shape=(self.input_dim,),name='GAN_input')
        GANmid1 = En(GANin)
        GANmid2 = De(GANmid1)
        D.trainable = False
        GANout = D(GANmid2)
        if(self.alpha!=None):
            aux_in = Kl.Input(shape=(1,),name=('aux_input'))
            GAN = Km.Model([GANin,aux_in],[GANout,GANmid2],name='GAN_AE')
        else:
            GAN = Km.Model(GANin,GANout,name='GAN_AE')
        
        # Custom loss function
        if(self.alpha==None):
            full_loss = partial(_full_loss,in_true=GANin,in_reco=GANmid2,eps=self.epsilon)
            GAN.compile(loss=full_loss,optimizer=Kopt.Adam(lr=0.0002,beta_1=0.5))
        else:
            #full_loss = partial(_full_loss_DisCo,in_true=GANin,in_reco=GANmid2,eps=self.epsilon,alph=self.alpha,var_1=aux_in,power=self.power)
            full_loss = partial(_full_loss,in_true=GANin,in_reco=GANmid2,eps=self.epsilon)
            full_loss.__name__ = 'full_loss'
            
            var_2 = tf.sqrt(tf.reduce_sum(tf.square(GANmid2 - GANin),axis=1))
            full_loss2 = partial(DisCo,var_1=aux_in,var_2=var_2,power=self.power,alph=self.alpha)
            full_loss2.__name__ = 'full_loss2'
            
            GAN.compile(loss=[full_loss,full_loss2],optimizer=Kopt.Adam(lr=0.0002,beta_1=0.5))
        
        #GAN.compile(loss=full_loss,optimizer=Kopt.Adam(lr=0.0002,beta_1=0.5))
        
        # Display the sumary if required
        if(display==True):
            En.summary()
            print('')
            De.summary()
            print('')
            AE.summary()
            print('')
            D.summary()
            print('')
            GAN.summary()
        
        return En,AE,D,GAN
    
    # Train one GAN-AE model
    def train(self,train_data,val_data,aux_data=None,ih=-1):
        '''
        Train a single GAN-AE model according to the hyperparameters defined for this instance.
        
        Arguments :
            train_data : The training dataset given as a numpy array.
            
            val_data : The validation dataset given as a numpy array. This dataset is used to
                       Evaluate the FoM at each training cycle.
            
            ih : Specify if the ethod is called from a separate thread. This argument is used only
                 when training multiple GAN-AE models simutaneously (it is then called from the
                 multi_train method).
                 If you call this method directly to train a single GAN-AE, please leave it by default.
        
        Note :
            All the given dataset must be properly scaled between 0 and 1.
            This can be done using the scale_data method before the training.
        '''
        
        # Check aux_data
        if(self.alpha!=None and aux_data==None):
            print('ERROR : You must specifify a auxilary variable for the DisCo term.')
            return
        
        # Function to train the discriminator only
        def D_train(append=True):
            D.tranable = True
            
            # Create a fake dataset using the AE
            train_fake = G.predict(train_data)
            train_full = np.concatenate((train_data,train_fake))
            
            val_fake = G.predict(val_data)
            val_full = np.concatenate((val_data,val_fake))
            
            # Create labels for D
            label_train_full = np.concatenate((np.ones(train_data.shape[0]),np.zeros(train_fake.shape[0])))
            label_val_full = np.concatenate((np.ones(val_data.shape[0]),np.zeros(val_fake.shape[0])))
            
            # Train D
            D.fit(x=train_full,y=label_train_full,
                  batch_size=self.batch_size,epochs=self.ND,verbose=0,
                  validation_data=(val_full,label_val_full),shuffle=True
                 )
            
            # Evaluate D and append results if required
            if(append):
                res_D = D.evaluate(x=train_full,y=label_train_full,batch_size=self.batch_size,verbose=0)
                res_D2 = D.evaluate(x=val_full,y=label_val_full,batch_size=self.batch_size,verbose=0)
                
                loss['D_train'].append(res_D)
                loss['D_val'].append(res_D2)
            
            return
        
        # Function to train the full GAN-AE (with D frozen)
        def GAN_train():
            D.trainable = False
            
            # Train GAN-AE
            if(self.alpha==None):
                GAN.fit(x=train_data,y=np.ones(train_data.shape[0]),
                        batch_size=self.batch_size,epochs=self.NGAN,verbose=0,
                        validation_data=(val_data,np.ones(val_data.shape[0])),shuffle=True
                       )
            else:
                GAN.fit(x=[train_data,aux_data[0]],y=[np.ones(train_data.shape[0]),train_data],
                        batch_size=self.batch_size,epochs=self.NGAN,verbose=0,
                        validation_data=([val_data,aux_data[1]],[np.ones(val_data.shape[0]),val_data]),shuffle=True
                       )
            
            # Evaluate G and append loss
            res_G = G.evaluate(x=train_data,y=train_data,batch_size=self.batch_size,verbose=0)
            res_G2 = G.evaluate(x=val_data,y=val_data,batch_size=self.batch_size,verbose=0)
            
            loss['G_train'].append(res_G)
            loss['G_val'].append(res_G2)
            
            # Evaluate full GAN-AE and append loss
            if(self.alpha==None):
                res_GAN = GAN.evaluate(x=train_data,y=np.ones(train_data.shape[0]),batch_size=self.batch_size,verbose=0)
                res_GAN2 = GAN.evaluate(x=val_data,y=np.ones(val_data.shape[0]),batch_size=self.batch_size,verbose=0)
            else:
                res_GAN = GAN.evaluate(x=[train_data,aux_data[0]],
                                       y=[np.ones(train_data.shape[0]),train_data],batch_size=self.batch_size,verbose=0)
                res_GAN2 = GAN.evaluate(x=[val_data,aux_data[1]],
                                        y=[np.ones(val_data.shape[0]),val_data],batch_size=self.batch_size,verbose=0)
            
            loss['GAN_train'].append(res_GAN[0])
            loss['GAN_val'].append(res_GAN2[0])
            
            return
        
        # Function to evaluate the model with the FoM
        def FoM_eval():
            # Evaluate the discriminator part
            F = D.predict(val_data)
            F = 1-F.mean()
            
            # Add the AE part
            F += loss['G_val'][-1]
            
            FoM.append(F)
            
            return
        
        # Initialize a tensorflow session
        sess = tf.Session()
        
        # Build the model
        if(ih==-1):
            En,G,D,GAN = self.build()
        else:
            En,G,D,GAN = self.build(display=False)
        
        # Check if should pretrain the AE part
        if(self.pretrain_AE):
            G.fit(x=train_data,y=train_data,
                  batch_size=self.batch_size,epochs=self.NGAN,verbose=0,
                  validation_data=(val_data,val_data),shuffle=True
                 )
        
        # Check if should pretrain the D part
        if(self.pretrain_dis):
            D_train(append=False)
        
        # Initilize loss containiner
        loss = dict()
        loss['G_train'] = []
        loss['G_val'] = []
        loss['D_train'] = []
        loss['D_val'] = []
        loss['GAN_train'] = []
        loss['GAN_val'] = []
        
        # Initilize FoM list
        FoM = []
        
        # Main cycle loop
        stop = int(0)
        best = 100.0
        for cyc in range(self.Ncycle):
            # D-only epochs
            D_train()
            
            # AE+D (with D forzen) epochs
            GAN_train()
            
            # Evaluation with FoM
            FoM_eval()
            
            # Check the early stop condition (if any)
            if(self.early_stop!=None):
                # Increment counter if no improvment
                if(FoM[-1]>=best):
                    stop += 1
                else:
                    best = FoM[-1]
                    stop = 0
                
                # Check if we should stop
                if(stop==self.early_stop):
                    #print('   stopping at cyc {} : {} ({})'.format(cyc,loss['GAN_train'][-1],FoM[-1]))
                    break
            #print('   cyc {} : {} ({})'.format(cyc,loss['GAN_train'][-1],FoM[-1]))
        
        # Convert all result containers in numpy array
        loss['G_train'] = np.array(loss['G_train'])
        loss['G_val'] = np.array(loss['G_val'])
        loss['D_train'] = np.array(loss['D_train'])
        loss['D_val'] = np.array(loss['D_val'])
        loss['GAN_train'] = np.array(loss['GAN_train'])
        loss['GAN_val'] = np.array(loss['GAN_val'])
        FoM = np.array(FoM)
        
        # Save results
        if(ih==-1):
            self.loss = loss
            self.FoM = FoM
            self.AE_weights = G.get_weights()
            self.dis_weights = D.get_weights()
        else:
            self.loss[ih] = loss
            self.FoM[ih] = FoM
            self.AE_weights[ih] = G.get_weights()
            self.dis_weights[ih] = D.get_weights()
        
        # Clear the tensorflow session
        sess.close()
        del sess
        
        return
    
    # Train many GAN-AE models and select the best ones
    def multi_train(self,train_data,val_data,aux_data=None):
        '''
        Train multiple GAN-AE models according to to the hyperparameters defined for
        this instance.
        The training of each individual models is parralelized and is done using the
        train method.
        
        Arguments :
            train_data : The training dataset given as a numpy array.
            
            val_data : The validation dataset given as a numpy array. This dataset is used to
                       Evaluate the FoM at each training cycle.
        
        Note :
            All the given dataset must be properly scaled between 0 and 1.
            This can be done using the scale_data method before the training.
        '''
        # Initialize result containers
        self.loss = np.empty(self.Nmodel,dtype=np.object)
        self.FoM = np.empty(self.Nmodel,dtype=np.object)
        self.AE_weights = np.empty(self.Nmodel,dtype=np.object)
        self.dis_weights = np.empty(self.Nmodel,dtype=np.object)
        
        # Thread pool executor
        if(self.Nworker>1):
            with thd.ThreadPoolExecutor(max_workers=self.Nworker) as exe:
                # Start all threads
                for th in range(self.Nmodel):
                    exe.submit(self.train,train_data,val_data,aux_data,th)
        else:
            for th in range(self.Nmodel):
                self.train(train_data,val_data,aux_data,th)
        
        # Retrieve the final FoM for each trained model
        FFoM = np.array([self.FoM[i][-1] for i in range(self.Nmodel)])
        
        # Sort the trained model by ascending FoM
        S = FFoM.argsort()
        self.FoM = self.FoM[S]
        self.loss = self.loss[S]
        self.AE_weights = self.AE_weights[S]
        self.dis_weights = self.dis_weights[S]
        
        return
    
    # Apply a single GAN-AE model
    # If label is given, they can be used to evaluate the model
    def apply(self,data,dmin,dmax, var_name=None,label=None,filename=None,do_latent=True,do_reco=True,do_distance=True,do_roc=True,do_auc=True,ih=-1):
        '''
        Apply one GAN-AE model to a dataset in order to produce result plots.
        
        Arguments :
            data : The dataset given as a numpy array.
            
            dmin : Numpy array specifying the true minimums of the input features.
                   This is used to plot the reconstructed data in real scale.
            
            dmax : Numpy array specifying the true maximums of the input features.
                   This is used to plot the reconstructed data in real scale.
            
            var_name : The x-axis labels to be used when ploting the original and/or reconstruted
                       features. If None, the names are built using the 'Var {i}' convention.
                       Defalut to None.
            
            label : Numpy array with truth label for the data.
                    The labels are used to separate the background and signal to compute the ROC
                    curve.
                    If None, the background and signal are not separated and no ROC curve is
                    computed.
                    Default to None.
            
            filename : The base name for the output files. This base is prepended to all the produced
                       files. 
                       If None, the plots are shown but not saved. Default to None.
            
            do_latent : Boolean specifying if the latent space should be ploted.
                        Default to True.
            
            do_reco : Boolean specifying if the reconstructed variables should be ploted.
                      Default to True.
            
            do_distance : Boolean specifying if the Euclidean distance distribution should be ploted.
                          The obtained distance distributions are recorded within this instance variables.
                          Default to True.
            
            do_roc : boolean specifying if the ROC curve should be ploted. This requires to give truth
                     labels.
                     Default to True.
            
            do_auc : Specify if the AUC should be recorded. Requires to give truth labels and to compute
                     the ROC curve.
                     Default to True.
        
            ih : Specify if the ethod is called from a separate thread. This argument is used only
                 when applying multiple GAN-AE models simutaneously (it is then called from the
                 multi_apply method).
                 If you call this method directly to apply a single GAN-AE, please leave it by default.
        
        Note :
            All the given dataset must be properly scaled between 0 and 1.
            This can be done using the scale_data method before the training.
        '''
        
        # Split the background and signal if labels given
        if(type(label)!=type(None)):
            bkg = data[label==0]
            sig = data[label==1]
        
        # Build the GAN-AE model and load the weights
        En,AE,D,GAN = self.build(display=False)
        if(ih==-1):
            AE.set_weights(self.AE_weights)
            D.set_weights(self.dis_weights)
        else:
            AE.set_weights(self.AE_weights[ih])
            D.set_weights(self.dis_weights[ih])
        
        # Latent space plot
        if(do_latent):
            # Apply the encoder model do the data
            if(type(label)==type(None)):
                cod_data = En.predict(data)
                Nplot = cod_data.shape[1]
            else:
                cod_bkg = En.predict(bkg)
                cod_sig = En.predict(sig)
                Nplot = cod_bkg.shape[1]
            
            # Do the plot
            with lock:
                F = plt.figure(figsize=(18,5*(Nplot//3+1)))
                plt.suptitle('Latent space distribution')
                for i in range(1,Nplot+1):
                    plt.subplot(3,Nplot//3+1,i)
                    if(type(label)==type(None)):
                        plt.hist(cod_data[:,i-1],bins=60,linewidth=2,histtype='step')
                    else:
                        plt.hist(cod_bkg[:,i-1],bins=60,linewidth=2,histtype='step',label='background')    
                        plt.hist(cod_sig[:,i-1],bins=60,linewidth=2,histtype='step',label='signal')
                        plt.legend()
                    plt.xlabel('latent {}'.format(i),size='large')
                if(filename==None):
                    plt.show()
                else:
                    if(ih==-1):
                        plt.savefig('{}_latent_space.png'.format(filename),bbox_inches='tight')
                    else:
                        plt.savefig('{0:s}_{1:2d}_latent_space.png'.format(filename,ih),bbox_inches='tight')
                    plt.close(F)
        
        # Reconstructed variables plot
        if(do_reco):
            # Apply the AE model do the data
            if(type(label)==type(None)):
                reco_data = AE.predict(data)
                Nplot = reco_data.shape[1]
                reco_data_restored = self.restore_data(reco_data,dmin,dmax)
                data_restored = self.restore_data(data,dmin,dmax)
            else:
                reco_bkg = AE.predict(bkg)
                reco_sig = AE.predict(sig)
                Nplot = reco_bkg.shape[1]
                reco_bkg_restored = self.restore_data(reco_bkg,dmin,dmax)
                reco_sig_restored = self.restore_data(reco_sig,dmin,dmax)
                bkg_restored = self.restore_data(bkg,dmin,dmax)
                sig_restored = self.restore_data(sig,dmin,dmax)
            
            # Check the original variable names
            if(type(var_name)==type(None)):
                var_name = np.array(['variable_{}'.format(i) for i in range(1,Nplot+1)])
            
            # Do the plot
            with lock:
                F = plt.figure(figsize=(18,5*(Nplot//3+1)))
                plt.suptitle('Reconstucted variable distribution')
                for i in range(1,Nplot+1):
                    plt.subplot(Nplot//3+1,3,i)
                    if(type(label)==type(None)):
                        plt.hist(reco_data_restored[:,i-1],bins=60,linewidth=2,color='b',histtype='step')
                        plt.hist(data_restored[:,i-1],bins=60,linewidth=2,color='b',linestyle='--',histtype='step')
                    else:
                        plt.hist(reco_bkg_restored[:,i-1],bins=60,linewidth=2,color='b',histtype='step',label='background')
                        plt.hist(bkg_restored[:,i-1],bins=60,linewidth=2,color='b',linestyle='--',histtype='step')
                        plt.hist(reco_sig_restored[:,i-1],bins=60,linewidth=2,color='r',histtype='step',label='signal')
                        plt.hist(sig_restored[:,i-1],bins=60,linewidth=2,color='r',linestyle='--',histtype='step')
                        plt.legend()
                    plt.xlabel(var_name[i-1])
                if(filename==None):
                    plt.show()
                else:
                    if(ih==-1):
                        plt.savefig('{}_reco_variables.png'.format(filename),bbox_inches='tight')
                    else:
                        plt.savefig('{0:s}_{1:2d}_reco_variables.png'.format(filename,ih),bbox_inches='tight')
                    plt.close(F)
        
        # Euclidean distance plot
        if(do_distance):
            # Check if we need to appy the AE model
            if(do_reco==False):
                if(type(label)==type(None)):
                    reco_data = AE.predict(data)
                else:
                    reco_bkg = AE.predict(bkg)
                    reco_sig = AE.predict(sig)
            
            # Compute the Euclidean distance distribution
            if(type(label)==type(None)):
                dist_data = np.sqrt(np.sum(np.square(data - reco_data),axis=1))
            else:
                dist_bkg = np.sqrt(np.sum(np.square(bkg - reco_bkg),axis=1))
                dist_sig = np.sqrt(np.sum(np.square(sig - reco_sig),axis=1))
            
            # Do the plot
            with lock:
                F = plt.figure(figsize=(12,8))
                plt.title('Euclidean distance distribution')
                if(type(label)==type(None)):
                    plt.hist(dist_data,bins=60,linewidth=2,histtype='step')
                else:
                    plt.hist(dist_bkg,bins=60,linewidth=2,histtype='step',label='background')
                    plt.hist(dist_sig,bins=60,linewidth=2,histtype='step',label='signal')
                    plt.legend(fontsize='large')
                plt.xlabel('Euclidean distance',size='large')
                if(filename==None):
                    plt.show()
                else:
                    if(ih==-1):
                        plt.savefig('{}_distance_distribution.png'.format(filename),bbox_inches='tight')
                    else:
                        plt.savefig('{0:s}_{1:2d}_distance_distribution.png'.format(filename,ih),bbox_inches='tight')
                    plt.close(F)
            
            # Save the best models
            if(ih==-1):
                if(type(label)==type(None)):
                    self.distance = dist_data
                else:
                    self.distance = [dist_bkg,dist_sig]
            else:
                if(type(label)==type(None)):
                    self.distance[:,ih] = dist_data
                else:
                    self.distance[0][:,ih] = dist_bkg
                    self.distance[1][:,ih] = dist_sig
        
        # ROC curve plot
        if(do_roc):
            # Check if the distance has been computed
            if(do_distance==False):
                print("CAN'T COMPUTE THE ROC CURVE !!")
                print("Please set 'do_distance=True'")
                return
            
            # Check if there are labels
            if(type(label)==type(None)):
                print("CAN'T COMPUTE THE ROC CURVE !!")
                print('Please give truth labels.')
                return
            
            # Now we can compute the roc curve
            Nbin = 100 # We use 100 points to make the ROC curve
            roc_min = min([dist_bkg.min(),dist_sig.min()])
            roc_max = max([dist_bkg.max(),dist_sig.max()])
            step = (roc_max-roc_min)/Nbin
            steps = np.arange(roc_min+step,roc_max+step,step)
            roc_x = np.array([dist_sig[dist_sig>th].size/dist_sig.size for th in steps])
            roc_y = np.array([dist_bkg[dist_bkg<th].size/dist_bkg.size for th in steps])
            roc_r1 = np.linspace(0,1,100)
            roc_r2 = 1-roc_r1
            
            # Compute AUC
            auc_sig = auc(roc_x,roc_y)
            
            # Do the plot
            with lock:
                F = plt.figure(figsize=(12,8))
                plt.plot(roc_x,roc_y,'-',label='auc={0:.4f}'.format(auc_sig))
                plt.plot(roc_r1,roc_r2,'--',label='random class')
                plt.legend(fontsize='large')
                plt.xlabel('signal efficiency',size='large')
                plt.ylabel('background rejection',size='large')
                if(filename==None):
                    plt.show()
                else:
                    if(ih==-1):
                        plt.savefig('{}_ROC_curve.png'.format(filename),bbox_inches='tight')
                    else:
                        plt.savefig('{0:s}_{1:2d}_ROC_curve.png'.format(filename,ih),bbox_inches='tight')
                    plt.close(F)
        
        # AUC distribution
        if(do_auc):
            # Check if ROC curve has been calculated
            if(do_roc==False):
                print("CAN'T COMPUTE AUC !!!")
                print("Please set 'do_roc=True'")
                return
            
            # Store the computed AUC
            if(ih==-1):
                self.auc = auc_sig
            else:
                self.auc[ih] = auc_sig
        
        return
    
    # Apply a many GAN-AE model and average them all
    # If label is given, use them to evaluate the models
    def multi_apply(self,data,dmin,dmax,var_name=None,label=None,filename=None,do_latent=True,do_reco=True,do_distance=True,do_roc=True,do_auc=True):
        '''
        Arguments :
            data : The dataset given as a numpy array.
            
            dmin : Numpy array specifying the true minimums of the input features.
                   This is used to plot the reconstructed data in real scale.
            
            dmax : Numpy array specifying the true maximums of the input features.
                   This is used to plot the reconstructed data in real scale.
            
            var_name : The x-axis labels to be used when ploting the original and/or reconstruted
                       features. If None, the names are built using the 'Var {i}' convention.
                       Defalut to None.
            
            label : Numpy array with truth label for the data.
                    The labels are used to separate the background and signal to compute the ROC
                    curve.
                    If None, the background and signal are not separated and no ROC curve is
                    computed.
                    Default to None.
            
            filename : The base name for the output files. This base is prepended to all the produced
                       files. For all the individual models, a unique id is appended to the base name.
                       If None, the plots are shown but not saved. Default to None.
            
            do_latent : Boolean specifying if the latent space should be ploted.
                        Default to True.
            
            do_reco : Boolean specifying if the reconstructed variables should be ploted.
                      Default to True.
            
            do_distance : Boolean specifying if the Euclidean distance distribution should be ploted.
                          The obtained distance distributions are recorded within this instance variables.
                          In addition, the averaged distance distribution is also ploted.
                          Default to True.
            
            do_roc : boolean specifying if the ROC curve should be ploted. This requires to give truth
                     labels.
                     In addition, the ROC curve obtained from the averaged distance distribution is also ploted.
                     Default to True.
            
            do_auc : Specify if the AUC should be recorded. Requires to give truth labels and to compute
                     the ROC curve.
                     In addition, a sccater plot showing the AUC versus FoM is also produced.
                     Default to True.
        
        Note :
            All the given dataset must be properly scaled between 0 and 1.
            This can be done using the scale_data method before the training.
        '''
        
        # Initialize result containers
        if(type(label)==type(None)):
            if(do_distance):
                self.distance = np.empty((data.shape[0],self.Nselec+1))
        else:
            if(do_distance):
                self.distance = [np.empty((label[label==0].size,self.Nselec+1)),np.empty((label[label==1].size,self.Nselec+1))]
            if(do_auc):
                self.auc = np.empty(self.Nselec+1)
        
        # Thread pool executor
        if(self.Nworker>1):
            with thd.ThreadPoolExecutor(max_workers=self.Nworker) as exe:
                for th in range(self.Nselec):
                    exe.submit(self.apply,data,dmin,dmax,var_name,label,filename,do_latent,do_reco,do_distance,do_roc,do_auc,th)
        else:
            for th in range(self.Nselec):
                self.apply(data,dmin,dmax,var_name,label,filename,do_latent,do_reco,do_distance,do_roc,do_auc,th)
        
        # Check if we should compute the distance
        if(do_distance):
            # Compute averaged distance
            if(type(label)==type(None)):
                dist_data_av = self.distance[:,:self.Nselec].mean(axis=1)
                self.distance[:,-1] = dist_data_av
            else:
                dist_bkg_av = self.distance[0][:,:self.Nselec].mean(axis=1)
                dist_sig_av = self.distance[1][:,:self.Nselec].mean(axis=1)
                self.distance[0][:,-1] = dist_bkg_av
                self.distance[1][:,-1] = dist_sig_av
            
            # Do the plot
            F = plt.figure(figsize=(12,8))
            plt.title('Averaged Eclidean distance distribution')
            if(type(label)==type(None)):
                plt.hist(dist_data_av,bins=60,linewidth=2,histtype='step')
            else:
                plt.hist(dist_bkg_av,bins=60,linewidth=2,histtype='step',label='background')
                plt.hist(dist_sig_av,bins=60,linewidth=2,histtype='step',label='signal')
                plt.legend(fontsize='large')
            plt.xlabel('Averaged Euclidean distance',size='large')
            if(filename==None):
                plt.show()
            else:
                plt.savefig('{}_distance_average.png'.format(filename),bbox_inches='tight')
                plt.close(F)
        
        if(do_roc):
            # Check if the distance has been computed
            if(do_distance==False):
                print("CAN'T COMPUTE THE ROC CURVE !!")
                print("Please set 'do_distance=True'")
                return
            
            # Check if there are labels
            if(type(label)==type(None)):
                print("CAN'T COMPUTE THE ROC CURVE !!")
                print('Please give truth labels.')
                return
            
            # Now we can compute the roc curve
            Nbin = 100 # We use 100 points to make the ROC curve
            roc_min = min([dist_bkg_av.min(),dist_sig_av.min()])
            roc_max = max([dist_bkg_av.max(),dist_sig_av.max()])
            step = (roc_max-roc_min)/Nbin
            steps = np.arange(roc_min+step,roc_max+step,step)
            roc_x = np.array([dist_sig_av[dist_sig_av>th].size/dist_sig_av.size for th in steps])
            roc_y = np.array([dist_bkg_av[dist_bkg_av<th].size/dist_bkg_av.size for th in steps])
            roc_r1 = np.linspace(0,1,100)
            roc_r2 = 1-roc_r1
            
            # Compute AUC
            auc_sig = auc(roc_x,roc_y)
            
            # Do the plot
            F = plt.figure(figsize=(12,8))
            plt.plot(roc_x,roc_y,'-',label='auc={0:.4f}'.format(auc_sig))
            plt.plot(roc_r1,roc_r2,'--',label='random class')
            plt.legend(fontsize='large')
            plt.xlabel('signal efficiency',size='large')
            plt.ylabel('background rejection',size='large')
            if(filename==None):
                plt.show()
            else:
                plt.savefig('{0:s}_ROC_curve.png'.format(filename),bbox_inches='tight')
                plt.close(F)
        
        # Check if we should do the AUC plot
        if(do_auc and type(label)!=type(None)):
            # Get the best FoM array.
            FFoM = np.array([self.FoM[i][-1] for i in range(self.Nselec)])
            
            # Save the global AUC value
            self.auc[-1] = auc_sig
            #print('    FFoM={}'.format(FFoM))
            #print('    auc={}'.format(self.auc))
            
            # Do a scatter plot FoM vs AUC
            F = plt.figure(figsize=(8,8))
            plt.title('FoM versus AUC')
            plt.scatter(FFoM,self.auc[:-1])
            plt.xlabel('FoM',size='large')
            plt.ylabel('AUC',size='large')
            if(filename==None):
                plt.show()
            else:
                plt.savefig('{}_scatter_AUC.png'.format(filename),bbox_inches='tight')
                plt.close(F)
        
        return
    
    # Save the trained models
    def save(self,filename):
        '''
        Save all the parameters and all the trained weights of this GAN-AE instance.
        The parameters are stored in a text file and the wights of each AE and D
        models are stored in HDF format.
        
        Arguments :
            filename : The base name of the save files. The full name of the files is
                       constructed from this base.
        '''
        
        # Save the parameters, loss and FoM
        Fparam = open('{}_param.txt'.format(filename),'w')
        param = dict()
        param['input_dim'] = self.input_dim
        param['hidden_dim'] = self.hidden_dim
        param['latent_dim'] = self.latent_dim
        param['dis_dim'] = self.dis_dim
        param['epsilon'] = self.epsilon
        param['alpha'] = self.alpha
        param['power'] = self.power
        param['NGAN'] = self.NGAN
        param['ND'] = self.ND
        param['batch_size'] = self.batch_size
        param['Ncycle'] = self.Ncycle
        param['early_stop'] = self.early_stop
        param['pretrain_AE'] = self.pretrain_AE
        param['pretrain_dis'] = self.pretrain_dis
        param['Nmodel'] = self.Nmodel
        param['Nselec'] = self.Nselec
        param['Nworker'] = self.Nworker
        
        # Check if there is no model to save
        if(self.AE_weights==[]):
            print('NO TRAINED GAN-AE MODEL FOUND !!')
            
            # Write the parameter file
            param['trained'] = 'None'
            print(param,file=Fparam)
            Fparam.close()
            return
        
        # Build the model
        En,AE,D,GAN = self.build(display=False)
        
        # Check if there is only one trained model to save
        if(type(self.loss)==type(dict())):
            print('    saving a single model')
            # Write the parameter file
            param['trained'] = 'single'
            print(param,file=Fparam)
            Fparam.close()
            
            # Set model weights
            AE.set_weights(self.AE_weights)
            D.set_weights(self.dis_weights)
            
            # Save the weights in HDF format
            AE.save_weights('{}_AE.h5'.format(filename))
            D.save_weights('{}_D.h5'.format(filename))
            
            # Save the loss and FoM in a separate file
            Floss = open('{}_loss.txt'.format(filename),'w')
            loss = dict()
            for l in self.loss:
                loss[l] = list(self.loss[l])
            loss['FoM'] = list(self.FoM)
            print(loss,file=Floss)
            Floss.close()
            return
        
        # Here, we have many models to save
        # Write the parameter file
        param['trained'] = 'multi'
        print(param,file=Fparam)
        Fparam.close()
        
        # Loop over all models
        for i in range(self.Nmodel):
            # Set model weights
            AE.set_weights(self.AE_weights[i])
            D.set_weights(self.dis_weights[i])
            
            # Save the weights in HDF format
            AE.save_weights('{0:s}_AE{1:2d}.h5'.format(filename,i))
            D.save_weights('{0:s}_D{1:2d}.h5'.format(filename,i))
            
            # Save the loss and FoM for model i in a separate file
            Floss = open('{0:s}_{1:2d}_loss.txt'.format(filename,i),'w')
            loss = dict()
            for l in self.loss[i]:
                loss[l] = list(self.loss[i][l])
            loss['FoM'] = list(self.FoM[i])
            print(loss,file=Floss)
            Floss.close()
        
        return
    
    # Load models from files
    def load(self,filename):
        '''
        Load the parameters and weights from a set of files.
        
        Arguments : 
            filename : Base name of the files. The full name of all the files is reconstructed
                       from this base.
        '''
        
        # First, load the parameters
        Fparam = open('{}_param.txt'.format(filename),'r')
        param = eval(Fparam.read())
        Fparam.close()
        
        # Restore the parameters
        self.input_dim = param['input_dim']
        self.hidden_dim = param['hidden_dim']
        self.latent_dim = param['latent_dim']
        self.dis_dim = param['dis_dim']
        self.epsilon = param['epsilon']
        self.NGAN = param['NGAN']
        self.ND = param['ND']
        self.batch_size = param['batch_size']
        self.Ncycle = param['Ncycle']
        self.early_stop = param['early_stop']
        self.pretran_AE = param['pretrain_AE']
        self.pretrain_dis = param['pretrain_dis']
        self.Nmodel = param['Nmodel']
        self.Nselec = param['Nselec']
        self.Nworker = param['Nworker']
        
        # Ensure retro-compatibility for previous versions saves
        # For that we load the new parameters only if they are present
        if('alpha' in param.keys()):
            self.alpha = param['alpha']
            self.power = param['power']
        else: # If they are not, we set them to default
            self.alpha = None
            self.power = 1
        
        # Check how many wieghts files we should load
        if(param['trained']=='None'):
            # No weights file
            print('NO WEIGHTS FILE AVAILABLE !!')
            
            # Initialize results
            self.FoM = []
            self.AE_weights = []
            self.dis_weights = []
            self.loss = []
            self.distance = []
            self.auc = []
            
        elif(param['trained']=='single'):
            # Only one model (so 2 weights files)
            En,AE,D,GAN = self.build(display=False)
            AE.load_weights('{}_AE.h5'.format(filename))
            D.load_weights('{}_D.h5'.format(filename))
            self.AE_weights = AE.get_weights()
            self.dis_weights = D.get_weights()
            
            # Also one loss file
            Floss = open('{}_loss.txt'.format(filename),'r')
            loss = eval(Floss.read())
            self.loss = dict()
            for l in loss:
                if l=='FoM':
                    self.FoM = np.array(loss[l])
                else:
                    self.loss[l] = np.array(loss[l])
            print(type(self.loss))
            self.distance = []
            self.auc = []
            
        else:
            # Nmodel models (so 2*Nmodel weights files)
            self.AE_weights = np.empty(self.Nmodel,dtype=np.object)
            self.dis_weights = np.empty(self.Nmodel,dtype=np.object)
            self.loss = np.empty(self.Nmodel,dtype=np.object)
            self.FoM = np.empty(self.Nmodel,dtype=np.object)
            En,AE,D,GAN = self.build(display=False)
            for i in range(self.Nmodel):
                AE.load_weights('{0:s}_AE{1:2d}.h5'.format(filename,i))
                D.load_weights('{0:s}_D{1:2d}.h5'.format(filename,i))
                self.AE_weights[i] = AE.get_weights()
                self.dis_weights[i] = D.get_weights()
                
                # Also the corresponding loss file
                Floss = open('{0:s}_{1:2d}_loss.txt'.format(filename,i),'r')
                loss = eval(Floss.read())
                self.loss[i] = dict()
                for l in loss:
                    if l=='FoM':
                        self.FoM[i] = np.array(loss[l])
                    else:
                        self.loss[i][l] = np.array(loss[l])
            
            self.distance = []
            self.auc = []
        
        return
    
    # Plot loss and FoM
    def plot(self,todo='best',filename=None,AEloss=True,Dloss=True,GANloss=True,FoM=True):
        '''
        Plot the loss curves and FoM curves of the trainend model.
        This method behave differently if several models have been trained.
        
        Arguments :
            todo : Specify the number of plots that are required. Ignored if only one model
                   have been trained.
                   Can be either 'all' if all models should be ploted or 'best' if only the
                   best models should be ploted.
                   Default to 'best'.
            
            filename : The base name of the save files. The full name of the files is
                       constructed from this base.
            
            AEloss : Booleaen spesifying if the AE loss should be plotted.
                     Default to True.
            
            Dloss : Booleaen spesifying if the discriminant loss should be plotted.
                    Default to True.
            
            GANloss : Booleaen spesifying if the full GAN-AE loss should be plotted.
                      Default to True.
            
            FoM : Booleaen spesifying if the Figure of Merit should be plotted.
                  Default to True.
            
        Note : 
            If several models have been trained, a id ranging from 0 to Nmodel-1 is appended to the base.
        '''
        
        # Do one plot
        def plot_one(y_train,y_val,name,title):
            # x-axis range
            x = np.arange(1,y_train.size+1,1)
            
            # Do the plot
            F = plt.figure(figsize=(12,8))
            plt.title(title)
            if(type(y_val)==type(None)):
                plt.plot(x,y_train, 'o-',linewidth=2)
            else:
                plt.plot(x,y_train, 'o-',linewidth=2,label='training')
                plt.plot(x,y_val, 'o-',linewidth=2,label='validation')
                plt.legend(fontsize='large')
            plt.xlabel('cycle',size='large')
            plt.ylabel(title,size='large')
            
            if(filename==None):
                plt.show()
            else:
                plt.savefig(name,bbox_inches='tight')
                plt.close(F)
            
            return
        
        # Check if several model have been trained
        if(type(self.loss)==type(dict())):
            # Here, only one model
            if(AEloss):
                plot_one(self.loss['G_train'],self.loss['G_val'],'{}_AE_loss.png'.format(filename),'AE loss (reco error)')
            if(Dloss):
                plot_one(self.loss['D_train'],self.loss['D_val'],'{}_discrim_loss.png'.format(filename),'discriminator loss (crossentropy)')
            if(GANloss):
                plot_one(self.loss['GAN_train'],self.loss['GAN_val'],'{}_full_loss.png'.format(filename),'full loss (combined)')
            if(FoM):
                plot_one(self.FoM,None,'{}_FoM.png'.format(filename),'Figure of Merit')
        else:
            # Here, sereral models
            if(todo=='best'):
                N = self.Nselec
            else:
                N = self.Nmodel
            
            for i in range(N):
                if(AEloss):
                    plot_one(self.loss[i]['G_train'],self.loss[i]['G_val'],'{0:s}_{1:2d}_AE_loss.png'.format(filename,i),'AE loss (reco error)')
                if(Dloss):
                    plot_one(self.loss[i]['D_train'],self.loss[i]['D_val'],'{0:s}_{1:2d}_discrim_loss.png'.format(filename,i),'discriminator loss (crossentropy)')
                if(GANloss):
                    plot_one(self.loss[i]['GAN_train'],self.loss[i]['GAN_val'],'{0:s}_{1:2d}_full_loss.png'.format(filename,i),'full loss (combined)')
                if(FoM):
                    plot_one(self.FoM[i],None,'{0:s}_{1:2d}_FoM.png'.format(filename,i),'Figure of Merit')
        
        return
    
    #end of GAN_AE class


