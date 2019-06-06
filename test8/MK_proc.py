'''
def det_bits(loaded,ind_det=24,nbits=7,verbose=True):
    # ME1/1 is ind_det=24
    # original binning is 260(about 8bits)
    import numpy as np
    bendmin = -(2**(nbits-1)-1)
    bendmax = 2**(nbits-1)-1
    X = loaded['variables']
    np.clip(X[:,ind_det],bendmin,bendmax,out=X[:,ind_det])
    if verbose == True:
        print("clip performed")
        print("max: ",np.nanmax(X[:,ind_det]))
        print("std: ",np.nanstd(X[:,ind_det]))
        print("min: ",np.nanmin(X[:,ind_det]))
    nan_ind = np.isnan(X[:,ind_det])
    two2n=2**nbits-2+1
    digi = np.linspace(bendmin,bendmax,num=two2n)
    if verbose == True:
        print(digi)
    yesnan = digi[np.digitize(X[:,ind_det],digi)-1] # new values for new number of bits 
    yesnan[nan_ind] = bendmax + 1 #np.nan # restore nan
    if verbose == True:
        print(X[0:100:,ind_det])
        print(yesnan[0:100])
    X[:,ind_det] = yesnan
    return X
'''

def det_bits(loaded,ind_det=24,nbits=7,verbose=True):
    # ME1/1 is ind_det=24
    # original binning is 260(about 8bits)
    import numpy as np
    X = loaded['variables']
    dictBins = {}
    # 4-bit compressions
    dictBins[41] = np.asarray([0,3,8,12,16,20,24,28]) # 5-bit truncated
    dictBins[42] = np.asarray([0,1,2,3, 5, 9, 17,33]) # Power-of-2
    dictBins[43] = np.asarray([0,1,2,4, 6, 9, 14,22]) # Fibbonaci-ish
    dictBins[44] = np.asarray([0,1,3,5, 8, 12,18,27]) # Log-1.6
    
    # 3-bit compressions
    dictBins[31] = np.asarray([0,5,13,21]) # 4-bit truncated
    dictBins[32] = np.asarray([0,2,5, 17]) # Power-of-2 mode
    dictBins[33] = np.asarray([0,3,9, 22]) # Fibbonaci-ish mode
    dictBins[34] = np.asarray([0,3,8, 18]) # Log-1.6 mode
    
    bins = dictBins[nbits]
    binsNp = np.asarray(bins)
    nums = X[:,ind_det] # bend variable for ME1/1 (ind_det =24)
    numsSign = nums.copy()
    numsSign[numsSign >= 0] = 1
    numsSign[numsSign < 0] = -1
    numsAbs = np.abs(nums)
    bin_ind = np.digitize(numsAbs,binsNp)
    result = binsNp[bin_ind-1]
    result = result * numsSign
    if verbose == True:
        print("clip performed")
        print("max: ",np.nanmax(result))
        print("std: ",np.nanstd(result))
        print("min: ",np.nanmin(result))
    nanStd = np.nanstd(result)
    nan_ind = np.isnan(X[:,ind_det]) # indicies for nan values
    if verbose == True:
        print(bins)

    result[nan_ind] = max(bins) + 1 # value for np.nan 
    if verbose == True:
        print(X[0:100:,ind_det])
        print(result[0:100])
    X[:,ind_det] = result
    return X, nanStd