import math
import matplotlib.pyplot as plt
import os
import numpy as np

my_absolute_dirpath = os.getcwd()
my_absolute_dirpath.replace('\\' ,'/')
my_absolute_dirpath = my_absolute_dirpath.replace('\\' ,'/')

# La = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]
# P = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# K = [-20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

L = list(np.arange(0, 185, 5, dtype=int))
P = list(np.arange(1, 21, 1, dtype=int))
K = list(np.arange(-20, 20, 1, dtype=int))


#%%

M = []

M1 = np.genfromtxt(my_absolute_dirpath + '/assets/map_1.csv', delimiter=',',  dtype = 'int')
M2 = np.genfromtxt(my_absolute_dirpath + '/assets/map_2.csv', delimiter=',',  dtype ='int')
M3 = np.genfromtxt(my_absolute_dirpath + '/assets/map_2.csv', delimiter=',',  dtype ='int').T



#%%


# for cord in listOfCoordinates:
#     print(cord)
#%%

# arr = np.array(M1)
# result = np.where(arr >= 0)
# listOfCoordinates = list(zip(result[1], result[0]))

# M_resh = list(np.reshape(M1, (np.shape(M1)[0]*np.shape(M1)[1], )))    



#%%
def iterateCells(M):
    
    arr = np.array(M)
    result = np.where(arr >= 0)
    listOfCoordinates = list(zip(result[1], result[0]))
    
    M_resh = np.reshape(M, (np.shape(M)[0]*np.shape(M)[1], ))
    
    Ro = []
    
    param = 0
    
    for cell, coords in zip(list(M_resh), listOfCoordinates):
        
        if cell == 1:
            
            vals = computeRo(coords[0]+param, coords[1]+param)
                  
            Ro.append(vals)
        
    Ro = np.reshape(np.array(Ro), (len(L), len(Ro)))
    Ro = list(Ro)
    Ro = [l.tolist() for l in Ro]        
    return Ro
        
    

           
def computeRo(x, y):
    # angle = 0
    
    vals = []
    
    for angle in L:
        
        R = x * math.cos(math.radians(angle)) + y * math.sin(math.radians(angle))
        
        if R > 0:
            R += 0.5
        elif R < 0:
            R -= 0.5
            
        Ro_val = int(R)

        vals.append(Ro_val)
    return vals           

def HjuSpectrum(Ro):
    
    H_spectrum = []
    
    summ = 0
    for angle in range(0, len(L)):
        
        for n in range(-40, 2):
            summ += Ro[angle].count(n)*Ro[angle].count(n)
        
        H_spectrum.append(summ)
        summ = 0
        #print(sum)
    return  H_spectrum



def spectrNorm(spectrum):
    maxSk = max(spectrum)
    minimum = min(spectrum)
    
    spectrNorm = []
    i = 0   
    for e in spectrum:
        spectrNorm.append(-(spectrum[i]-minimum)/(maxSk-minimum))
        i += 1
        
    maximum = max(spectrNorm)
    return list(np.array(spectrNorm)-maximum+1)

def spectrNorm2(spectrum):
    maxSk = max(spectrum)
    minimum = min(spectrum)
    
    spectrNorm = []
    i = 0   
    for e in spectrum:
        spectrNorm.append(-(spectrum[i]-minimum)/(maxSk-minimum))
        i += 1
        

    return spectrNorm
        

def correlation(spectrum_1, spectrum_2):
    summ = 0
    n = 0
    
    
    corr = []
    for e in spectrum_1:
        i = 0
        for e in spectrum_1:
            if i+n > len(spectrum_1)-1:
                summ += spectrum_1[i]*spectrum_2[i+n-len(spectrum_2)]

            else:
                summ += spectrum_1[i]*spectrum_2[i+n]

            i += 1
        corr.append(summ)
        summ = 0
        n += 1
    return corr

def XYspectr(M):
    
    X_out = []
    Y_out = []
    
    Y = 0
    X = 0
    for y in range(0, len(P)):
        for x in range(0, len(P)):
            if M[y][x] == 1:
                Y += 1
        Y_out.append(Y)
        Y = 0

    for x in range(0, len(P)):
        for y in range(0, len(P)):
            if M[y][x] == 1:
                X += 1
        X_out.append(X)
        X = 0
    
    return X_out, Y_out


def XYcorr_1(Xsp1, Ysp1, Xsp2, Ysp2):
    
    Xcorr = []
    Ycorr = []
    
    summ = 0
    n = 20
    for e in Xsp1:
        i = 0
        for e in Xsp1:
            if i+n > (len(K)/2)-1:
                pass

            else:
                summ += Xsp1[i]*Xsp2[i+n]

            i += 1
        Xcorr.append(summ)
        summ = 0
        n -= 1

    summ = 0
    n = 20
    for e in Ysp1:
        i = 0
        for e in Ysp1:
            if i+n > (len(K)/2)-1:
                pass

            else:
                summ += Ysp1[i]*Ysp2[i+n]

            i += 1
        Ycorr.append(summ)
        summ = 0
        n -= 1
        
    summ = 0
    n = 0
    for e in Xsp1:
        i = 0
        for e in Xsp1:
            if i+n > (len(K)/2)-1:
                pass

            else:
                summ += Xsp1[i]*Xsp2[i+n]

            i += 1
        Xcorr.append(summ)
        summ = 0
        n += 1

    summ = 0
    n = 0
    for e in Ysp1:
        i = 0
        for e in Ysp1:
            if i+n > (len(K)/2)-1:
                pass

            else:
                summ += Ysp1[i]*Ysp2[i+n]

            i += 1
        Ycorr.append(summ)
        summ = 0
        n += 1        
    
    return Xcorr, Ycorr



## M1
Ro_M1 = iterateCells(M1)
Spectrum_M1 = HjuSpectrum(Ro_M1)
Spectrum_M1_norm = spectrNorm(Spectrum_M1)

Spectrum_M1_stacked = np.vstack([np.array(L), np.array(Spectrum_M1)]).T
Spectrum_M1_norm_stacked = np.vstack([np.array(L), np.array(Spectrum_M1_norm)]).T

# plt.plot(np.array(L), np.array(Spectrum_M1_norm))
# plt.grid()




np.savetxt(my_absolute_dirpath + '/out/Spectrum_M1.csv', Spectrum_M1, delimiter=",")
np.savetxt(my_absolute_dirpath + '/out/Spectrum_M1_norm.csv', Spectrum_M1_norm_stacked, delimiter=",")

## M2
Ro_M2 = iterateCells(M2)
Spectrum_M2 = HjuSpectrum(Ro_M2)
Spectrum_M2_norm = spectrNorm(Spectrum_M2)

Spectrum_M2_stacked = np.vstack([np.array(L), np.array(Spectrum_M2)]).T
Spectrum_M2_norm_stacked = np.vstack([np.array(L), np.array(Spectrum_M2_norm)]).T

# plt.plot(np.array(L), np.array(Spectrum_M2_norm))
# plt.grid()

np.savetxt(my_absolute_dirpath + '/out/Spectrum_M2.csv', Spectrum_M2, delimiter=",")
np.savetxt(my_absolute_dirpath + '/out/Spectrum_M2_norm.csv', Spectrum_M2_norm_stacked, delimiter=",")

## Correlation

corr_M1_M2 = correlation(Spectrum_M1_norm, Spectrum_M2_norm)
corr_M1_M2_norm = spectrNorm(corr_M1_M2)

plt.plot(np.array(L), np.array(corr_M1_M2_norm))
plt.grid()

corr_M1_M2_stacked = np.vstack([np.array(L), np.array(corr_M1_M2)]).T
corr_M1_M2_norm_stacked = np.vstack([np.array(L), np.array(corr_M1_M2_norm)]).T

# np.savetxt(my_absolute_dirpath + '/out/corr_M1_M2.csv', corr_M1_M2_stacked, delimiter=",")
np.savetxt(my_absolute_dirpath + '/out/corr_M1_M2_norm.csv', corr_M1_M2_norm_stacked, delimiter=",")


## XY corr

Xsp1, Ysp1 = XYspectr(M1)
Xsp2, Ysp2 = XYspectr(M3)

Xsp1_norm = spectrNorm(Xsp1)
Ysp1_norm = spectrNorm(Ysp1)
Xsp2_norm = spectrNorm(Xsp2)
Ysp2_norm = spectrNorm(Ysp2)

Xsp1_norm_stacked = np.vstack([np.array(P), np.array(Xsp1_norm)]).T
# np.savetxt(my_absolute_dirpath + '/out/Xsp1_norm_stacked.csv', Xsp1_norm_stacked, delimiter=",")

Ysp1_norm_stacked = np.vstack([np.array(P), np.array(Ysp1_norm)]).T
# np.savetxt(my_absolute_dirpath + '/out/Ysp1_norm_stacked.csv', Ysp1_norm_stacked, delimiter=",")

Xsp2_norm_stacked = np.vstack([np.array(P), np.array(Xsp2_norm)]).T
# np.savetxt(my_absolute_dirpath + '/out/Xsp2_norm_stacked.csv', Xsp2_norm_stacked, delimiter=",")

Ysp2_norm_stacked = np.vstack([np.array(P), np.array(Ysp2_norm)]).T
# np.savetxt(my_absolute_dirpath + '/out/Ysp2_norm_stacked.csv', Ysp2_norm_stacked, delimiter=",")


Xcorr, Ycorr = XYcorr_1(Xsp1, Ysp1, Xsp2, Ysp2)

Xcorr_norm = spectrNorm(Xcorr)
Ycorr_norm = spectrNorm(Ycorr)

Xcorr_stacked = np.vstack([np.array(K), np.array(Xcorr)]).T
Ycorr_stacked = np.vstack([np.array(K), np.array(Ycorr)]).T

np.savetxt(my_absolute_dirpath + '/out/Xcorr_stacked.csv', Xcorr_stacked, delimiter=",")
np.savetxt(my_absolute_dirpath + '/out/Ycorr_stacked.csv', Ycorr_stacked, delimiter=",")

maxXsp = max(Xcorr)
maxYsp = max(Ycorr)

print("Maximum value, X translation:",K[Xcorr.index(maxXsp)], ", Y translation:", K[Ycorr.index(maxYsp)])
print("Corr max values: X =",maxXsp, ", Y =", maxYsp)


