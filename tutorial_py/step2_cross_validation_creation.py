import numpy as np
import os
import cv2
import random
import pickle
from step1_patch_extraction import struct
from step1_patch_extraction import subfilestruct
from sklearn.model_selection import KFold

nucleus_dir = '../DL_tutorial_Code/1-nuclei/'

def make_training_list():

    with open(nucleus_dir + 'array.pickle', 'rb') as f:
        patient_struct = pickle.load(f)
    nfolds = 5
    fidtrain = []
    fidtest = []

    fidtrain_parent = []
    fidtest_parent = []
    for zz in range(1,nfolds+1):
        fidtrain.insert(zz, open(nucleus_dir+ 'train_w32_'+str(zz)+'.txt', 'w'))
        fidtest.insert(zz, open(nucleus_dir+ 'test_w32_'+str(zz)+'.txt', 'w'))

        fidtrain_parent.insert(zz, open(nucleus_dir+ 'train_w32_parent_'+str(zz)+'.txt', 'w'))
        fidtest_parent.insert(zz, open(nucleus_dir+ 'test_w32_parent_'+str(zz)+'.txt', 'w'))

    kf = KFold(n_splits = nfolds, shuffle = True)
    k = 0

    for train, test in kf.split(patient_struct):
        print(train, test)
        for fi in range(len(patient_struct)):
            if fi in test:
                fid = fidtest[k]
                fid_parent = fidtest_parent[k]
            else:
                fid = fidtrain[k]
                fid_parent = fidtrain_parent[k]
            fid_parent.write(patient_struct[fi].base + '\n')

            subfiles = patient_struct[fi].sub_file

            for subfi in range(len(subfiles)):
                try:
                    subfnames = subfiles[subfi].fnames_subs_neg
                    for zz in range(len(subfnames)):
                        subfname = subfnames[zz]
                        fid.write(' 0\n'.join(subfname))
                        fid.write(' 0\n')

                    subfnames = subfiles[subfi].fnames_subs_pos
                    for zz in range(len(subfnames)):
                        subfname = subfnames[zz]
                        fid.write(' 1\n'.join(subfname))
                        fid.write(' 1\n')
                except:
                    print('error')
                    print([patient_struct[fi].base, patient_struct[fi].subfile[subfi].base])
        k+=1




    for zz in range(nfolds):
        fidtrain[zz].close()
        fidtest[zz].close()

        fidtrain_parent[zz].close()
        fidtest_parent[zz].close()




if __name__ == '__main__':


    make_training_list()
