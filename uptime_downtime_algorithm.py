import numpy as np
from sklearn.model_selection import train_test_split
import pywt
import scipy.signal



class Up_down_time():
    def __init__(self,model):
        self.model=model
        
    def image_compression_spectogram(self,data,sample_rate,wavelet='db1',keep=1.):
        f, t, Sxx = scipy.signal.spectrogram(data, sample_rate)
        coeffs = pywt.wavedec2(Sxx,wavelet=wavelet,level=4)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        Csort = np.sort(np.abs(coeff_arr.reshape(-1)))
        thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
        ind = np.abs(coeff_arr) > thresh
        Cfilt = coeff_arr * ind        
        coeffs_filt = pywt.array_to_coeffs(Cfilt,coeff_slices,output_format='wavedec2')      
        Arecon = pywt.waverec2(coeffs_filt,wavelet=wavelet)
    
        return Arecon
    
    def training(self,train_data,label_train_data,time_train_data,test_size=0.25,seed=123):
        samples=[]
        for sample,label,T in zip(train_data,label_train_data,time_train_data):
            sample_rate=len(sample)/T
            spec_compressed=self.image_compression_spectogram(sample,sample_rate)
            spec_compressed=np.transpose(spec_compressed)
            rows=(spec_compressed.shape)[0]
            samples.append(np.hstack((spec_compressed,np.ones((rows,1))*label)))
        
        X_Y=samples[0]
        for i in range(1,len(samples)):
            X_Y=np.vstack((X_Y,samples[i]))

        
        np.random.seed(seed)        
        X_train,X_test,Y_train,Y_test=train_test_split(X_Y[:,:-1],X_Y[:,-1],test_size=test_size,stratify=X_Y[:,-1])
        
        self.model.fit(X_train,Y_train)
        accuracy=self.model.score(X_test,Y_test)

        test_result='It was used %d seconds to train the model\nIt was used %d seconds to test the model\nThe test result obtained an accuracy of  %.2f %%' %(len(Y_train),len(Y_test),accuracy*100) 
        print(test_result)
        
    def run(self,data,time_data):
        time_on=0
        time_off=0
        for sample,dt in zip(data,time_data):
            sample_rate=len(sample)/dt
            img_compressed=self.image_compression_spectogram(sample,sample_rate)
            X=np.transpose(img_compressed)
            Y=self.model.predict(X)
            time_on+=sum(Y) 
            time_off+=sum(Y-1)*-1
        minutes_on=time_on//60
        seconds_on=time_on%60
        minutes_off=time_off//60
        seconds_off=time_off%60

        time='%d minutes and %d seconds ON\n%d minutes and %d seconds OFF' % (minutes_on,seconds_on,minutes_off,seconds_off)
        
        print(time)
        
        
    
