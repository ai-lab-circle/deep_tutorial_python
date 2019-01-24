# Nucleui segmentation (in progress)
  
This source includes nucleui segmentation code using deep learing framework(caffe).   
We converted matlab code[2] to python code to run without MATLAB and implemented on ubuntu 16.04, python3.5, and caffe.

## Requirments
python3.5  
[caffe](https://github.com/BVLC/caffe.git)


## Overview  
### Data  
Download [datasets](http://andrewjanowczyk.com/wp-static/nuclei.tgz)
```bash
   tar -xvzf nuclei.tgz
   mv nuclei/*  ~/public/DL_tutorial_Code/1-nuclei/images
```

### Pre-trained model and prototxt files  
```bash
   cd DL_tutorial_code/1-nuclei/models/
   cd DL_tutorial_code/common/
```


### Training/Testing  
```bash
   cd tuturial_py/
   step1_patch_extraction.py 
   step2_cross_validation_creation.py (training and testing list creation step)
   step3_make_db.py (database creation step)
   step4_submit_jobs.py (training step)
   step5_create_output_images_kfold.py (testing step)
```

## Acknowledgements  
 We would like to thank the authors of DLtutorialCode[2], which we use in this work.

## References  
[1]Janowczyk, A., Madabhushi, A., 2016. Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases. Journal of Pathology Informatics 7, 29.   
[2][original source](https://github.com/choosehappy/public/tree/master/DL%20tutorial%20Code)



