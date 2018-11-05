# Accent Embeddings - Multitask Learning
This repository store a part of the work done as mentioned in the paper submitted in Interspeech 2018.
[Paper](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/1864.html "IS1864")

Pre-Requisites - 
1. You have worked with the Kaldi toolkit and are quite familiar with it, meaning you are familiar with training a DNN Acoustic Model and know the requirements.
2. We use Mozilla CommonVoice Dataset for all the experiments. A detailed split can be found at - 
[Accents Unearthed](https://sites.google.com/view/accentsunearthed-dhvani/ "AccentsUnearthed")

## What we are doing?
To start with, we are training a standard Multitask Learning(MTL) Network where the primary task (T1 hereon) is the triphone state recognition and the secondary task (T2 hereon) is the accent classification. At the end, we use the T1 network as our DNN Acoustic Model for Speech Recognition.
This is the script [multitask_run_2_base_2.sh](./multitask_run_2_base_2.sh).

## Data Prep
1. We adapted the scripts using the babel-multilang setup where they train a multilingual system using multiple languages.
2. As we have 2 tasks, so the data folder should contain 2 directories, one for each task.
3. Each of these directories will contain the training as well as testing data in Kaldi format.
4. We named the directories - 101-recognition for T1, and 102-class for T2. 101-recogntion contained the training set cv-train-nz as well as all the dev and test sets. 102-class contained the same training set with a slight variation, cv-trainx-nz, this is contains the same utterances as cv-train-nz but each utterance-id is appended with an 'x' just to make the utterance ids from cv-train-nz and cv-trainx-nz different. Kaldi requires this for MTL.

**Note :** In the scripts,
1. Train7 - cv_train_nz, cv_trainx_nz
2. Dev4 - cv_dev_nz
3. Test4 - cv_test_nz
4. TestNZ - cv_test_onlynz
5. TestIN - cv_test_onlyindian

## Steps
Rest of the steps are pretty standard.
1. **MFCCS** - The script creates both standard and hires MFCCS of train, dev and test data provided the correct paths. It can also create speed perturbed data and its MFCCs (used in training).
2. **ivectors** - The script assumes a pretrained ivector extractor. The scripts for training an ivector extractor comes with kaldi.
3. **alignments** - For MTL, two types of alignments are required. The alignments for **T1** are the standard triphone state label sequences. The script can do that for you. It assumes a pre-trained GMM-HMM model to create those. Generating the alignments of **T2** is somewhat typical. To summarize, one needs accent label for every frame of each utterance in kaldi table format indexed by utterance ids. Look at [temp.cc](./temp.cc)
4. Once all these are done, xconfig and training is standard.
5. We use the T1 network for final decoding.


**Note :** The rest of the scripts are minor variations of this said script where we modify the network architecture.
