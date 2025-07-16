import numpy as np
from numba import jit,njit,prange,get_num_threads,set_num_threads,numba
import pickle
import matplotlib.pyplot as plt

def main(nsp,q,mp,vtc,ppc,dx,dy,nx,ny,dt,nt,vb,isav,isavp,run):
    nop=ppc*nx*ny  #total number of particles
    lx=dx*nx       #simulation box size x-axis
    ly=dy*ny       #simulation box size y-axis

    up=np.zeros((nsp,2,nop)) #particle position
    vp=np.zeros((nsp,2,nop)) #particle velocity

    ### set initial condition ###
    for isp in range(nsp):
        if(isp==0):
            up[isp,0,:]=np.random.rand(nop)*lx
            up[isp,1,:]=np.random.rand(nop)*ly
        else:
            up[isp,0,:]=up[0,0,:]
            up[isp,1,:]=up[0,1,:]

    for isp in range(nsp):
        vp[isp,0,:]=np.random.normal(0,vtc[isp],nop)#+vb[isp]
        vp[isp,1,:]=np.random.normal(0,vtc[isp],nop)#+vb[isp]

    plt.plot(up[0,0,:],up[0,1,:],'.')
    plt.show()

    plt.hist(vp[0,0,:],100)
    plt.show()
    
    ### set Fourier space
    kx=2*np.pi*(np.fft.fftfreq(nx,dx))#2*mt.pi/lx*np.r_[np.arange(nx/2),np.arange(-nx/2,0)]
    ky=2*np.pi*(np.fft.fftfreq(ny,dy))#2*mt.pi/ly*np.r_[np.arange(ny/2),np.arange(-ny/2,0)]
    KX,KY=np.meshgrid(kx,ky)
    #KF=np.ones((ny,nx))#np.exp(-36*((np.abs(KX)/np.max(KX))**36+(np.abs(KY)/np.max(KY))**36))
    
#    ### iteration starts ###
#    for it in range(nt+1):
#        up=push(nsp,nop,up,vp,lx,ly,0.5*dt)
#        
#        ds=dens(ppc,nsp,dx,dy,nx,ny,nop,q,up)
#
#        ex,ey=field(ds,KX,KY,KF)
#
#        vp=acc(nsp,q,mp,dx,dy,nx,ny,nop,up,vp,ex,ey,dt)
#
#        up=push(nsp,nop,up,vp,lx,ly,0.5*dt)
#        
#        if(it%isav==0 or it%isavp==0):
#            output(run,it,isav,isavp,up,vp,ds,ex,ey)  #output data
#        
#        if(it%100==0):
#            ### energy conservation check ###
#            eneptcl1= 0.5*mp[0]*np.sum((vp[0,0,:]**2+vp[0,1,:]**2)/ppc*dx*dy)
#            eneptcl2= 0.5*mp[1]*np.sum((vp[1,0,:]**2+vp[1,1,:]**2)/ppc*dx*dy)
#            eneel   = 0.5*np.sum(ex**2+ey**2)*dx*dy
#            print('t=',it*dt,'etot=',eneptcl1+eneptcl2+eneel)

    #### save parameters ###
    del up, vp, ds, ex, ey
    local_vars=locals()
    with open('data/%s_param.pkl' %(run), 'wb') as file:
        pickle.dump(local_vars, file)
        
    return None

#def push(nsp,nop,up,vp,lx,ly,dt):
#    
#    return up
#
#def acc(nsp,q,mp,dx,dy,nx,ny,nop,up,vp,ex,ey,dt):
#
#    return vp
#
#def dens(ppc,nsp,dx,dy,nx,ny,nop,q,up):
#
#    return ds
#
#def field(ds,KX,KY,KF):
#    
#    return ex,ey

def output(run,it,isav,isavp,up,vp,ds,ex,ey):
    if(it%isav==0):
        iout=it//isav
        data={'ds':ds,'ex':ex, 'ey':ey}
        with open('data/%s_fld%05d.pkl' %(run,iout), 'wb') as file:
            pickle.dump(data, file)
            
    if(it%isavp==0):
        iout=it//isavp
        data={'up':up,'vp':vp}
        with open('data/%s_ptcl%05d.pkl' %(run,iout), 'wb') as file:
            pickle.dump(data, file)
