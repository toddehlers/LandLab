"""
These are two helper functions which are needed for setting up the
climate oscillations in landlab

created by: Manuel Schmid, December 2017
"""

import numpy as np

##---OSCILLATING---#
#This function creates a asymetrical wave-form for rainfall input
def createAsymWave(baseLevel,
                   posAmp,
                   negAmp,
                   period,
                   time,
                   dt):
    #This sets up the basic sin-waveform which underlies the algorith
    baseTime = np.arange(0,time,dt)
    print('created Timeline')
    baseFreq = (2 * np.pi) / period
    print('established Frequency')
    baseWave = np.sin(baseFreq * baseTime) + (baseLevel)
    print('created waveform')
    
    #now we create a wave with different amplitudes by vector addition
    #positive Part
    posInd = np.where(baseWave > baseLevel)
    posSinAmp = posAmp - baseLevel
    posSinWave = posSinAmp * np.sin(baseFreq * baseTime) + baseLevel
    baseWave[posInd] = posSinWave[posInd]
    #negative Part
    negInd = np.where(baseWave < baseLevel)
    negSinAmp = baseLevel - negAmp
    negSinWave = negSinAmp * np.sin(baseFreq * baseTime) + baseLevel
    baseWave[negInd] = negSinWave[negInd]

    return baseWave
    
def createSymWave(baseLevel,
                 ampMax,
                 ampMin,
                 period,
                 time,
                 dt):
    baseTime = np.arange(0,time,dt)
    print('created Timeline')
    baseFreq = (2 * np.pi) / period
    print('established Frequency')
    ampTotal = np.abs((ampMax - ampMin) / 2)
    newBase = np.abs((ampMax + ampMin) / 2)
    baseWave =  ampTotal * np.sin(baseFreq * baseTime) + newBase
    print('created waveform')
    
    return baseWave
