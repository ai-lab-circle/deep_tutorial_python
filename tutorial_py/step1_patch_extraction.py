import numpy as np
import os
import cv2
import random
import threading
import time
import pickle

##@@@  user local directory
input_dir = '../DL_tutorial_Code/1-nuclei/images/'
output_dir = '../DL_tutorial_Code/1-nuclei/subs/'
nucleus_dir = '../DL_tutorial_Code/1-nuclei/'
#write_names = []
class subfilestruct:
    def __init__(self, base, fnames_subs_pos, fnames_subs_neg):
        self.base = base
        self.fnames_subs_pos = fnames_subs_pos
        self.fnames_subs_neg = fnames_subs_neg
class struct:
    def __init__(self, base):
        self.base = base
        self.sub_file = []
    def add_subfile(self, idx, base, fnames_subs_pos, fnames_subs_neg):
        s = subfilestruct(base, fnames_subs_pos, fnames_subs_neg)
        self.sub_file.insert(idx, s)

def makeNegativeMask(io, patchDim):
    ior = io[:, :, 2]
    _, img = cv2.threshold(ior, 100, 255, cv2.THRESH_TRUNC)
    _, img2 = cv2.threshold(ior, 75, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((6, 6), np.uint8)
    bw = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    mask = np.zeros(((io.shape[0]+2), (io.shape[1]+2)), np.uint8)
    cv2.floodFill(bw, mask, (0, 0), 0)

    kernel = np.ones((20, 20), np.uint8)
    bw = cv2.dilate(bw, kernel, 3)

    kernel = np.ones((patchDim, patchDim), np.float32)/(patchDim*patchDim)
    im = cv2.filter2D(bw, -1, kernel)
    bw - cv2.subtract(bw, im)


    bw = cv2.bitwise_not(bw)

    return bw

def makeNegativeMask_dilate_sub(io, dilate_size):
    io_orig = io
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    io = cv2.dilate(io, kernel, 1)
    o = io - io_orig

    return o

def func_extraction_worker_w_rots(outdir, base, io, hwsize, r, c, max_num, cla, prefix, rots):
    image_size = io.shape
    toremove = [1 if (r[i]+2*hwsize) >image_size[0] or (c[i]+2*hwsize) > image_size[1] or (r[i]-2*hwsize) < 1 or (c[i]-2*hwsize) < 1 else 0 for i in range(len(r))]
    print('remove '+str(sum(toremove)))
    Rremoved = [r[i] for i in range(len(r)) if toremove[i] != 1]
    Cremoved = [c[i] for i in range(len(c)) if toremove[i] != 1]
    idx = [i for i in range(len(Rremoved))]
    random.shuffle(idx)
    Rremoved = [Rremoved[i] for i in idx]
    Cremoved = [Cremoved[i] for i in idx]
    ni = min(len(Rremoved), round(max_num))

    write_names = []
    threads = []
    for ri in range(ni):
        my_thread = threading.Thread(target = parfor, args = (ri, io, Rremoved, hwsize, Cremoved, rots, base, cla, prefix, outdir, write_names))
        my_thread.start()
        threads.append(my_thread)
    for th in threads:
        th.join()

    return write_names

def parfor(ri, io, Rremoved, hwsize, Cremoved, rots, base, cla, prefix, outdir, write_names):
    patch = io[int(Rremoved[ri]-2*hwsize):int(Rremoved[ri]+2*hwsize-1), int(Cremoved[ri]-2*hwsize):int(Cremoved[ri]+2*hwsize-1), :]
    w, h, _ = patch.shape
    write_names_I = []
    for roti in range(len(rots)):
        degr = rots[roti]
        rotate = cv2.getRotationMatrix2D((w/2, h/2), degr, 1)
        rpatch = cv2.warpAffine(patch, rotate, (w, h))
        [nrow, ncol, ndim] = rpatch.shape
        rpatch = rpatch[int(nrow/2-hwsize):int(nrow/2+hwsize-1), int(ncol/2-hwsize):int(ncol/2+hwsize-1), :]
        pname = base+'_'+str(cla)+'_'+prefix+'_'+str(ri)+'_'+str(degr)+'.png'
        write_names_I.insert(roti, pname)
        cv2.imwrite(os.path.join(outdir, pname), rpatch)

    write_names.insert(ri, write_names_I)

def make_patches():
    outdir = output_dir
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    wsize = 32
    hwsize = wsize/2

    ratio_nuclei_dilation = 1
    ration_background = .3
    npositive_sample_per_image = 2500

    patients = [f.split('_')[0] for f in os.listdir(input_dir) if f.endswith('_mask.png')]
    patients = list(set(patients))
    patient_struct = []

    print(len(patients))
    for ci in range(len(patients)):
        print('patient: ', ci)
        p = struct(patients[ci])
        patient_struct.append(p)

        files = [f for f in os.listdir(input_dir) if f.startswith(patients[ci]+'_') and f.endswith('.tif')]

        for fi in range(len(files)):
            print([ci, len(patients), fi, len(files)])
            fname = files[fi]
            image_base = patients[ci]+'_'+str(fi)
            fname_mask = fname.replace('_original.tif', '_mask.png')



            try:
                io = cv2.imread(input_dir +fname, cv2.IMREAD_COLOR)
                io_mask = cv2.imread(input_dir+fname_mask, cv2.IMREAD_GRAYSCALE)
                io_sub = []

                for i in range(3):
                    div1 = io[:, :, i]
                    io_sub.append(np.pad(div1, [wsize, wsize], 'symmetric'))
                io = np.dstack([io_sub[0], io_sub[1], io_sub[2]])
                io_mask = np.pad(io_mask, [wsize, wsize], 'symmetric')
                print(io_mask.shape)
            except:
                print("error")
                continue
            (r, c)= io_mask.nonzero()
            r = r.tolist()
            c = c.tolist()
            fnames_subs_pos = func_extraction_worker_w_rots(outdir, image_base, io, hwsize, r, c, npositive_sample_per_image, 1, 'p', [0, 90])
            npos = len(fnames_subs_pos)
            (r, c) = makeNegativeMask_dilate_sub(io_mask, 3).nonzero()
            r = r.tolist()
            c = c.tolist()
            fnames_subs_neg1 = func_extraction_worker_w_rots(outdir, image_base, io, hwsize, r, c, npos*ratio_nuclei_dilation, 0, 'e', [0, 90])
            (r, c) = makeNegativeMask(io, 50).nonzero()
            r = r.tolist()
            c = c.tolist()
            fnames_subs_neg2 = func_extraction_worker_w_rots(outdir, image_base, io, hwsize, r, c, npos*ration_background, 0, 'b', [0, 90])
            fnames_subs_neg = fnames_subs_neg1 + fnames_subs_neg2
            patient_struct[ci].add_subfile(fi, fname, fnames_subs_pos, fnames_subs_neg)
    with open(nucleus_dir + 'array.pickle', 'wb') as f:
        pickle.dump(patient_struct, f)




if __name__ == '__main__':
    make_patches()


