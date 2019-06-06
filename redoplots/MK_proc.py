def det_bits(loaded,ind_det=24,nbits=7,verbose=True):
    # ME1/1 is ind_det=24
    # original binning is 260(about 8bits)
    import numpy as np
    bendmin = -(2**(nbits-1)-1)
    bendmax = 2**(nbits-1)-1
    X = loaded['variables']
    if verbose == True:
        print(X[0:20:,ind_det])
    np.clip(X[:,ind_det],bendmin,bendmax,out=X[:,ind_det])
    if verbose == True:
        print("clip performed")
        print("max: ",np.nanmax(X[:,ind_det]))
        print("min: ",np.nanmin(X[:,ind_det]))
    nanlist = np.isnan(X[:,ind_det])
    two2n=2**nbits-2+1
    digi = np.linspace(bendmin,bendmax,num=two2n)
    if verbose == True:
        print(digi)
    yesnan = digi[np.digitize(X[:,ind_det],digi)-1] # new values for new number of bits 
    yesnan[nanlist] = np.nan # restore nan
    if verbose == True:
        print(yesnan[0:20])
    X[:,ind_det] = yesnan
    return X