import numpy as np


def autocorrelate (a):

    b=np.concatenate((a,np.zeros(len(a))),axis=0)
    c= np.fft.ifft(np.fft.fft(b)*np.conjugate(np.fft.fft(b))).real
    d=c[:int(len(c)/2)]
    d=d/(np.array(range(len(a)))+1)[::-1]
    return d


