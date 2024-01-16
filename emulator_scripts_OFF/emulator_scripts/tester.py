import scipy.io 
import numpy as np 
import sys
import glob

files = sorted(glob.glob('*.mat'))

for file in files:

    print(file)

    a=(np.mean(scipy.io.loadmat(file)['nsocial'][0][0]))
    b=(len(scipy.io.loadmat(file)['nsocial'][0][0][0]))
    a1=(np.mean(scipy.io.loadmat(file)['nsocial'][0][1]))
    b1=(len(scipy.io.loadmat(file)['nsocial'][0][1][0]))
    c=(np.mean(scipy.io.loadmat(file)['nsocial'][1][0]))
    d=(len(scipy.io.loadmat(file)['nsocial'][1][0][0]))
    c1=(np.mean(scipy.io.loadmat(file)['nsocial'][1][1]))
    d1=(len(scipy.io.loadmat(file)['nsocial'][1][1][0]))
    e=(np.mean(scipy.io.loadmat(file)['nsocial'][2][0]))
    f=(len(scipy.io.loadmat(file)['nsocial'][2][0][0]))
    e1=(np.mean(scipy.io.loadmat(file)['nsocial'][2][1]))
    f1=(len(scipy.io.loadmat(file)['nsocial'][2][1][0]))

    print('nsocial native, true: ',a1, b1,'; fake:', a, b)
    print('nsocial social,  true: ', c1, d1,'; fake ',c, d)
    print('nsocial emusocial, true: ', e1, f1,' ; fake:', e, f)

    print('\n------------------ \n')

    a=(np.mean(scipy.io.loadmat(file)['social'][0][0]))
    b=(len(scipy.io.loadmat(file)['social'][0][0][0]))
    a1=(np.mean(scipy.io.loadmat(file)['social'][0][1]))
    b1=(len(scipy.io.loadmat(file)['social'][0][1][0]))
    c=(np.mean(scipy.io.loadmat(file)['social'][1][0]))
    d=(len(scipy.io.loadmat(file)['social'][1][0][0]))
    c1=(np.mean(scipy.io.loadmat(file)['social'][1][1]))
    d1=(len(scipy.io.loadmat(file)['social'][1][1][0]))
    e=(np.mean(scipy.io.loadmat(file)['social'][2][0]))
    f=(len(scipy.io.loadmat(file)['social'][2][0][0]))
    e1=(np.mean(scipy.io.loadmat(file)['social'][2][1]))
    f1=(len(scipy.io.loadmat(file)['social'][2][1][0]))

    print('social native, true: ',a1, b1,'; fake:', a, b)
    print('social social,  true: ', c1, d1,'; fake ',c, d)
    print('social emusocial, true: ', e1, f1,' ; fake:', e, f)

    print('\n------------------ \n')

    a=(np.mean(scipy.io.loadmat(file)['emu_social'][0][0]))
    b=(len(scipy.io.loadmat(file)['emu_social'][0][0][0]))
    a1=(np.mean(scipy.io.loadmat(file)['emu_social'][0][1]))
    b1=(len(scipy.io.loadmat(file)['emu_social'][0][1][0]))
    c=(np.mean(scipy.io.loadmat(file)['emu_social'][1][0]))
    d=(len(scipy.io.loadmat(file)['emu_social'][1][0][0]))
    c1=(np.mean(scipy.io.loadmat(file)['emu_social'][1][1]))
    d1=(len(scipy.io.loadmat(file)['emu_social'][1][1][0]))
    e=(np.mean(scipy.io.loadmat(file)['emu_social'][2][0]))
    f=(len(scipy.io.loadmat(file)['emu_social'][2][0][0]))
    e1=(np.mean(scipy.io.loadmat(file)['emu_social'][2][1]))
    f1=(len(scipy.io.loadmat(file)['emu_social'][2][1][0]))

    print('emusocial native, true: ',a1, b1,'; fake:', a, b)
    print('emusocial social,  true: ', c1, d1,'; fake ',c, d)
    print('emusocial emusocial, true: ', e1, f1,' ; fake:', e, f)

    print('\n####################\n')
    print('\n####################\n')
