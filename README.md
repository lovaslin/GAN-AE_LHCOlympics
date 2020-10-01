# GAN-AE_LHCOlympics
Code used for the LHC Olympics 2020 challenge.

The code used for the preprocessing can be found here :  
https://gitlab.cern.ch/idinu/lhc-olympics-preprocessing

The preprocessed data files can be found here :  
https://onedrive.live.com/?authkey=%21APNzARJylhUVxt0&id=2C3CDD05B333D5E2%214457&cid=2C3CDD05B333D5E2

### To use the code

The code requires python>=3.6 to run.

First, install all the dependencies (I recommend doing that in a new virtual environement) :
```bash
pip install requirement.txt

Dowload the preprocessed data and put them in a `data` folder.

```
To train the model :
```bash
python train.py
```

To apply the trained model and the bump hunting algorithm :
```bash
python apply.py
```

