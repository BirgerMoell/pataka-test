
import torch
import scipy.io.wavfile as wav
import json
import webrtcvad
import sys
from transformers import AutoProcessor, AutoModelForCTC
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import os

def init():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
    return device,processor,model


# vadlen = frame length in sec must be 0.01, 0.02 or 0.03
def getvad(fs,y,vadlen):

    assert(fs==16000)
    vad = webrtcvad.Vad(3)
    i = 0
    vaddata = []
    #tvad = []
    
    while True:
        vadframe = y[i:i+int(fs*vadlen)].tobytes()
        if len(vadframe)<2*fs*vadlen:
            break
        
        vaddata.append(vad.is_speech(vadframe,fs))
        #tvad.append(i/fs)
        i += int(fs*vadlen)
    xvad = np.array(vaddata).astype(np.int16)
    xvad[0:7]=0
    #plt.plot(np.arange(xvad.size)*vadlen,xvad,'.')
    #plt.plot(np.arange(y.size)/fs,y/(2**15))
    #plt.show()

    return xvad
   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)


def recognize(fs,y,device,processor,model,xvad=None):
    
    input_values = processor(y.astype('float32'), sampling_rate=fs, return_tensors="pt").input_values.to(device)
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    output = processor.batch_decode(predicted_ids[0],output_char_offsets=True)
    recframes = output['char_offsets']
    nframes = len(recframes)

    rate=50
    currentphone = ''
    phoneseq = []
    if xvad is not None:
        vadlist = xvad.tolist()
    else:
        vadlist = [1]*nframes
    
    for rec,vad,count in zip(recframes,vadlist,range(nframes)):
        token=''
        for xx in rec:
            token += xx['char']
        if token != '<pad>':
            currentphone = token
        if vad:
            phoneseq.append(currentphone)
        else:
            phoneseq.append('sil')

    nphoneseq = []
    currentphone = ''
    for phone in reversed(phoneseq):
        if phone != '':
            currentphone = phone
        else:
            phone = currentphone
        nphoneseq.append(phone)
    nphoneseq.reverse()
    phoneseq = nphoneseq
         
    lastphone = ''
    labels = []
    label = []
    for i,phone in enumerate(phoneseq):
        if phone != lastphone:
            #print(i,phone,phones)
            #import pdb;pdb.set_trace()
            if label:
                label.append(i/rate) #end time
                label.append(lastphone) # phone
                labels.append(label)
            label = [i/rate]
        lastphone = phone
    return labels

if __name__ == '__main__':
    device,processor,model = init()
    
    annotdir = 'devset_annotations'

    for wavfile in sys.argv[1:]:
        print(wavfile)
        jsonfile = wavfile.replace('.wav','.json')
        jsonfile = os.path.join(annotdir,os.path.split(jsonfile)[1])
        labfile = wavfile.replace('.wav','.lab')
        print(wavfile,jsonfile,labfile)
        
        fs,y = wav.read(wavfile)
        y = y.astype(np.int16)
        assert(fs==16000)
        assert(len(y.shape)==1)

        vadlen = 0.02
        if os.path.exists(jsonfile):
            # read speaker turns from json file and convert to vad
            turns = json.loads(open(jsonfile).read())
            xvad = np.zeros(int((y.shape[0]/fs)/vadlen))
            for clip in turns['clips']:
                i0 = int(clip['start_time']/vadlen)
                i1 = int(clip['end_time']/vadlen)
                xvad[i0:i1]=1
        else:
            print('could not find turnfile',jsonfile,'using webrtcvad instead')
            xvad = getvad(fs,y,vadlen)

        #        import pdb;pdb.set_trace()

        #    phones = recognize(fs,y,device,processor,model)
        phones = recognize(fs,y,device,processor,model,xvad)
        
        print(y.size/fs,'s')

        #json.dump({'tiers':{'phones':{'entries':phones}}},open(jsonfile,'w'))
        with open(labfile,'w') as f:
            for p in phones:
                t0,t1,sym = p
                f.write('{}\t{}\t{}\n'.format(t0,t1,sym))