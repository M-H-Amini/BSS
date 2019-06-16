# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:48:49 2019

@author: MHA
"""

from scipy.io import wavfile
import numpy.linalg as alg
import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt

class SoundMixer:
    def __init__(self,sound_paths):
        self.sound_paths=sound_paths
        self.sounds=[wavfile.read(sound_paths[i]) for i in range(len(sound_paths))]
        self.samp_freq=self.sounds[0][0]
    
    def getSoundsMatrix(self):
        self.sounds_matrix=np.zeros((1,len(self.sounds[0][1])))
        for i in range(len(self.sounds)):
            self.sounds_matrix=np.concatenate((self.sounds_matrix,np.reshape(self.sounds[i][1],(1,-1))))
        self.sounds_matrix=self.sounds_matrix[1:,:]
        return self.sounds_matrix
    
    def amplifySound(self,sound_index, db):
        sound_samp_freq=self.sounds[sound_index][0]
        sound_frames=self.sounds[sound_index][1]
        result=(sound_frames/2.**15)*10**(db/20)
        return (sound_samp_freq,result)
    
    def mixSounds(self,sound_indexes,dbs,file_name=False):
        new_sounds=[self.amplifySound(i,dbs[i]) for i in sound_indexes]
        mixed_sound_frames=sum([i[1] for i in new_sounds])
        if file_name:
            wavfile.write(file_name,self.samp_freq,mixed_sound_frames)
        return self.samp_freq,mixed_sound_frames
     
    def writeSound(self, sound, file_name):
        wavfile.write(file_name, sound[0], sound[1])
        
    def plotSound(self, sound, ranges, color='g'):
        data=sound[1][ranges[0]:ranges[1]]
        ranges=range(ranges[1]-ranges[0])
        plt.plot(ranges,data,color)
        
class BSS:
    def __init__(self,X, sampling_frequency):
        self.rawX=X 
        self.samp_freq=sampling_frequency
        
    def preprocess(self):
        ##  Centering...
        self.mean=np.reshape(np.mean(self.rawX,1),(-1,1))
        self.X=self.rawX-self.mean
        ##  Whitening...
        cov=np.cov(self.X)
        eigvals,eigvecs=alg.eig(cov)
        eigvals_prime=np.diag(1/(np.sqrt(eigvals)))
        eigvals=np.diag(eigvals)
        ##  Whitening transform...
        self.V=np.dot(eigvecs,np.dot(eigvals_prime,np.transpose(eigvecs)))
        self.X=np.dot(self.V,self.X)
    
    def estimateKurtosis(self,X):
        '''
        X is assumed to be a whitened single random variable...
        '''
        poweredX=np.power(X,4)
        result=np.mean(poweredX)-3
        return result
    
    def GramSchmidt(self,w):
        '''
        w must be a column vector...
        '''
        a=w  ##  A square matrix with all columns equal to w
        for i in range(w.shape[0]-1):
            a=np.concatenate((a,w),1)
        basis,_=alg.qr(a)
        return basis
    
    def recoverSources(self):
        '''
        Orthogonal matrix W and data X are used to find sources
        '''
        return np.dot(np.transpose(self.W),self.X)
    
    def writeSourcesToAudios(self,common_file_names):
        '''
        For example all files start with "ica"
        '''
        for i in range(self.S.shape[0]):
            wavfile.write('{}_{}.wav'.format(common_file_names,i+1), self.samp_freq, self.S[i,:])
    
    def orthogonalize(self,W):
        
        #ortho=np.dot(sp.linalg.fractional_matrix_power(np.linalg.inv(np.dot(W,np.transpose(W))),0.5),W)
        
        term1=np.dot(W,np.transpose(W))
        print('term1',term1)
        term2=alg.pinv(term1)
        print('term2',term2)
        term3=sp.linalg.fractional_matrix_power(term2,0.5)
        print('term3',term3)
        term4=np.dot(term3,W)
        ortho=term4
        #ortho=np.dot(sp.linalg.fractional_matrix_power(np.linalg.inv(np.dot(W,np.transpose(W))),0.5),W)
        #norms=alg.norm(ortho,axis=0)
        #ortho=ortho/norms
        
        print('********',ortho)
        #print('$$$$$$$$',norms)
        #print('%%%%%%%%',alg.norm(ortho,axis=0))
        return ortho
        
    def list2Matrix(self,w):
        W=np.zeros((len(w),1))
        for i in range(len(w)):
            W=np.concatenate((W,w[i]),axis=1)
        W=W[:,1:]
        return W
    
    def matrix2List(self,W):
        w=[]
        for i in range(W.shape[0]):
            w.append(W[:,i:i+1])
        return w
    
    def ICA_Gaussianity(self, learning_rate, max_iters=1000, details=False):
        t0=time.time()
        no_of_samples=self.X.shape[1]
        w=np.random.randn(self.X.shape[0],1)
        w/=alg.norm(w)
        sign=self.estimateKurtosis(np.dot(np.transpose(w),self.X))
        sign/=abs(sign)
        if details:
            print('sign is',sign)
        for i in range(max_iters):
            if details:
                if i/(max_iters/10) == int(i/(max_iters/10)):
                    print('{}% done...'.format(i/(max_iters/100)))
            if details:
                print('Iteration {}...'.format(i))
                print('Old w...',w)
            index=np.random.randint(0,no_of_samples)
            delta_w=sign*self.X[:,index:index+1]*(np.dot(np.transpose(w),self.X[:,index:index+1]))**3
            w+=learning_rate*delta_w
            w/=alg.norm(w)
            if details:
                print('New w...',w)
        if details:
            print('New w...',w)
        self.W=self.GramSchmidt(w)
        self.S=self.recoverSources()
        t1=time.time()
        if details:
            print('It took {} seconds!!!'.format(t1-t0))
        return self.S

    def __FastICA_Gaussianity(self, max_iters=10, tolerance=1e-8, initial_w=0, details=False):
        if type(initial_w)==int:
            w=np.random.randn(self.X.shape[0],1)
        else:
            w=initial_w
        w/=alg.norm(w)
        w=np.reshape(np.mean(np.power(np.dot(np.transpose(w),self.X),3)*self.X,1),(-1,1))-3*w
        w/=alg.norm(w)
        if details:
            print('New w...',w)
        return w
    
    
    def FastICA_Gaussianity(self, max_iters=10, tolerance=1e-8, initial_w=0, details=False):
        t0=time.time()
        W=[np.random.randn(self.X.shape[0],1) for i in range(self.X.shape[0])]
        W=[W[i]/np.linalg.norm(W[i]) for i in range(len(W))]
        self.W=self.list2Matrix(W)
        self.W=self.orthogonalize(self.W)
        iter_no=0
        while True:
            if details:
                print('*************Iteration {}**************'.format(iter_no))
            iter_no+=1
            oldSelfW=self.W
            W=self.matrix2List(self.W)
            for i in range(len(W)):
                W[i]=self.__FastICA_Gaussianity(max_iters, tolerance, W[i], False)
            self.W=self.list2Matrix(W)
            self.W=self.orthogonalize(self.W)
            #self.W=self.GramSchmidt(self.W[:,0:1])
            if np.allclose(oldSelfW,self.W,atol=tolerance) or iter_no==max_iters:
                if details:
                    print('Converged!!!')
                break
        self.S=self.recoverSources()
        t1=time.time()
        if details:
            print('W...',self.W)
            print('It took {} seconds!!!'.format(t1-t0))
        return self.S
    
    def __FastICA_Negentropy(self, G= lambda x: np.tanh(x), dG= lambda x: 1-np.power((np.tanh(x)),2), max_iters= 50, tolerance=1e-8, initial_w=0, details=False):
        if type(initial_w)==int:
            w=np.random.randn(self.X.shape[0],1)
        else:
            w=initial_w
        w/=alg.norm(w)
        part1=np.reshape(np.mean(self.X*G(np.dot(np.transpose(w),self.X)),1),(-1,1))
        part2=np.reshape(np.mean(dG(np.dot(np.transpose(w),self.X)),1),(-1,1))*w
        w=part1-part2
        w/=alg.norm(w)
        if details:
            print('New w...',w)
        return w
    
    def FastICA_Negentropy(self, G= lambda x: np.tanh(x), dG= lambda x: 1-np.power((np.tanh(x)),2), max_iters= 50, tolerance=1e-8, details=False):
        t0=time.time()
        W=[np.random.randn(self.X.shape[0],1) for i in range(self.X.shape[0])]
        W=[W[i]/np.linalg.norm(W[i]) for i in range(len(W))]
        self.W=self.list2Matrix(W)
        self.W=self.orthogonalize(self.W)
        iter_no=0
        while True:
            if details:
                print('*************Iteration {}**************'.format(iter_no))
            iter_no+=1
            oldSelfW=self.W
            W=self.matrix2List(self.W)
            for i in range(len(W)):
                W[i]=self.__FastICA_Negentropy(G, dG, max_iters, tolerance, W[i], False)
            self.W=self.list2Matrix(W)
            self.W=self.orthogonalize(self.W)
            #self.W=self.GramSchmidt(self.W[:,0:1])
            if np.allclose(oldSelfW,self.W,atol=tolerance) or iter_no==max_iters:
                if details:
                    print('Converged!!!')
                break
        self.S=self.recoverSources()
        t1=time.time()
        if details:
            print('W...',self.W)
            print('It took {} seconds!!!'.format(t1-t0))
        return self.S
        
if __name__=='__main__':
    ##  Generating sounds...
    plt.figure()
    sm=SoundMixer(['1.wav','2.wav','3.wav'])
    f1,mixed1=sm.mixSounds([0,1,2],[5,10,0],'mix1.wav')
    f2,mixed2=sm.mixSounds([0,1,2],[0,20,10],'mix2.wav')
    f3,mixed3=sm.mixSounds([0,1,2],[-10,5,20],'mix3.wav')
    mixed1=np.reshape(mixed1,(1,-1))
    mixed2=np.reshape(mixed2,(1,-1))
    mixed3=np.reshape(mixed3,(1,-1))
    mixed=np.concatenate((mixed1,mixed2,mixed3))
    #sm.plotSound(sm.sounds[0],[-1000,-1])
    #sm.plotSound(sm.sounds[1],[-1000,-1],'r')
    sm.plotSound((sm.samp_freq,mixed2[0,:]),[-1000,-1],'r')
    ##  BSS...
    bss=BSS(mixed,sm.samp_freq)
    bss.preprocess()
    #bss.ICA_Gaussianity(0.01,int(3e4),details=True)
    #bss.FastICA_Gaussianity(20,tolerance=-1, details=True)
    
    G1=lambda x: np.tanh(x)
    dG1=lambda x: 1-np.power((np.tanh(x)),2)
    G2=lambda x: x*np.exp(-np.power(x,2)/2)
    dG2=lambda x: (1-np.power(x,2))*np.exp(-np.power(x,2)/2)
    bss.FastICA_Negentropy(max_iters=20,details=True)
    #bss.FastICA_Negentropy(G=G2, dG= dG2, details=True)
    bss.writeSourcesToAudios('icas')
    sm.plotSound((sm.samp_freq,bss.S[0,:]),[-1000,-1])
    sm.getSoundsMatrix()
    '''
    w=np.random.randn(4,4)
    w=w/alg.norm(w,axis=0)
    print('wlll',w,alg.norm(w,axis=0))
    w=bss.orthogonalize(w)
    print('nwlll',w,alg.norm(w,axis=0))
    print(np.dot(np.transpose(w[:,0:1]),w[:,3:4]))
    '''
