import librosa
import numpy as np
from python_speech_features import fbank
import pickle
import os

def make_re(dir,name,file_name):
    def normalize_frames(m,Scale=True):
        if Scale:
            return(m-np.mean(m,axis=0))/(np.std(m,axis=0)+2e-12)
        else:
            return(m-np.mean(m,axis=0))

    
    filename=os.path.join(dir,name,file_name+'.wav')

    sample_rate=16000
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=40, winlen=0.025)
    filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
    feature = normalize_frames(filter_banks, Scale=False)       

    speaker_label=filename.split('\\')[-2]
    speaker_label=speaker_label[0:6]
    print(speaker_label)
    feat_and_label = {'feat':feature, 'label':speaker_label}

    filename_=filename.split('.')
    print(filename_[0])
    with open(filename_[0]+'.p', 'wb') as fp:
        pickle.dump(feat_and_label, fp)

    with open(filename_[0]+'.p','rb') as f:
        data=pickle.load(f)
   
if __name__ == '__main__':
    normalize_frames