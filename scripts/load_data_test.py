import tarfile
import numpy as np 

tar = tarfile.open("spdata.tar.gz", "r:gz")
for member in tar.getmembers():
    print(member)
    f = tar.extractfile(member)
    if f is not None:
        content = f.read()
        Data = np.loadtxt(content)
        print(Data)
        input('')

#region DUMP
"""
https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html
https://docs.python.org/2/library/tarfile.html
"""
#endregion



