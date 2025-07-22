import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math as mt
#from numba import jit,njit,prange,get_num_threads,set_num_threads,numba
from numba import jit 
import scipy.fftpack as sf
from IPython import display
import cProfile
import pickle

def main(nsp,q,mp,vtc,ppc,dx,dy,nx,ny,dt,nt,vb,isav,isavp,run):
    #np.seterr(divide='ignore', invalid='ignore')
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

    ##### Landau Damping ###
    #up[0,1,:]=np.random.rand(nop)*ly
    #up[1,0,:]=np.random.rand(nop)*lx
    #up[1,1,:]=np.random.rand(nop)*ly
    #A=0.01
    #kk=0.5
    #ymax=1+A
    #for ip in range(nop): 
    #    y_i= 9999
    #    z_i=-9999
    #    while y_i>z_i:
    #        x_i=np.random.rand()*lx
    #        y_i=ymax*np.random.rand()
    #        z_i=1+A*np.cos(kk*x_i)
    #    up[0,0,ip]=x_i

    for isp in range(nsp):
        vp[isp,0,:]=np.random.normal(0,vtc[isp],nop)+vb[isp]
        vp[isp,1,:]=np.random.normal(0,vtc[isp],nop)+vb[isp]
    
    ### set Fourier space
    kx=2*np.pi*(np.fft.fftfreq(nx,dx))#2*mt.pi/lx*np.r_[np.arange(nx/2),np.arange(-nx/2,0)]
    ky=2*np.pi*(np.fft.fftfreq(ny,dy))#2*mt.pi/ly*np.r_[np.arange(ny/2),np.arange(-ny/2,0)]
    KX,KY=np.meshgrid(kx,ky)
    if(ny==1): 
        KF=np.exp(-36*((np.abs(KX)/np.max(KX))**36))
    else:
        KF=np.exp(-36*((np.abs(KX)/np.max(KX))**36+(np.abs(KY)/np.max(KY))**36))
    
    ### time iteration starts ###
    for it in range(nt+1):
        up=push(nsp,nop,up,vp,lx,ly,0.5*dt)
        
        ds=dens(ppc,nsp,dx,dy,nx,ny,nop,q,up)

        ex,ey=field(ds,KX,KY,KF)

        vp=acc(nsp,q,mp,dx,dy,nx,ny,nop,up,vp,ex,ey,dt)

        up=push(nsp,nop,up,vp,lx,ly,0.5*dt)
        
        if(it%isav==0 or it%isavp==0):
            output(run,it,isav,isavp,up,vp,ds,ex,ey)
        
        if(it%100==0):
            ### energy conservation check ###
            eneptcl1= 0.5*mp[0]*np.sum((vp[0,0,:]**2+vp[0,1,:]**2)/ppc*dx*dy)
            eneptcl2= 0.5*mp[1]*np.sum((vp[1,0,:]**2+vp[1,1,:]**2)/ppc*dx*dy)
            eneel   = 0.5*np.sum(ex**2+ey**2)*dx*dy
            print('t=',it*dt,'etot=',eneptcl1+eneptcl2+eneel)
 
    #### save parameters ###
    del up, vp, ds, ex, ey
    local_vars=locals()
    with open('data/%s_param.pkl' %(run), 'wb') as file:
        pickle.dump(local_vars, file)
        
    return None

@jit()   
def push(nsp,nop,up,vp,lx,ly,dt):
    for isp in range(nsp):
        for ip in range(nop):
            up[isp,0,ip]+=dt*vp[isp,0,ip]
            up[isp,1,ip]+=dt*vp[isp,1,ip]
            
            ### boundary condition ###
            if(up[isp,0,ip]>lx): up[isp,0,ip]-=lx
            if(up[isp,0,ip]< 0): up[isp,0,ip]+=lx
            if(up[isp,1,ip]>ly): up[isp,1,ip]-=ly
            if(up[isp,1,ip]< 0): up[isp,1,ip]+=ly
    
    return up

@jit()
def acc(nsp,q,mp,dx,dy,nx,ny,nop,up,vp,ex,ey,dt):
    for isp in range(nsp):
        for ip in range(nop):
            ixm=mt.floor(up[isp,0,ip]/dx); ixp=ixm+1
            iym=mt.floor(up[isp,1,ip]/dy); iyp=iym+1
            wxp=up[isp,0,ip]/dx-ixm; wxm=1-wxp
            wyp=up[isp,1,ip]/dy-iym; wym=1-wyp
    
            ### boundary ###
            if ixp>nx-1: ixp=ixp-nx
            if iyp>ny-1: iyp=iyp-ny
    
            vp[isp,0,ip]+=q[isp]/mp[isp]*dt*(wym*wxm*ex[iym,ixm]+wyp*wxm*ex[iyp,ixm]+wym*wxp*ex[iym,ixp]+wyp*wxp*ex[iyp,ixp])
            vp[isp,1,ip]+=q[isp]/mp[isp]*dt*(wym*wxm*ey[iym,ixm]+wyp*wxm*ey[iyp,ixm]+wym*wxp*ey[iym,ixp]+wyp*wxp*ey[iyp,ixp])

    return vp

@jit()
def dens(ppc,nsp,dx,dy,nx,ny,nop,q,up):
    ds=np.zeros((nsp,ny,nx))
    for isp in range(nsp):
        for ip in range(nop):
            ixm=mt.floor(up[isp,0,ip]/dx); ixp=ixm+1
            iym=mt.floor(up[isp,1,ip]/dy); iyp=iym+1
            wxp=up[isp,0,ip]/dx-ixm; wxm=1-wxp
            wyp=up[isp,1,ip]/dy-iym; wym=1-wyp
    
            ### boundary ###
            if ixp>nx-1: ixp=ixp-nx
            if iyp>ny-1: iyp=iyp-ny
 
            ds[isp,iym,ixm]+=wym*wxm
            ds[isp,iym,ixp]+=wym*wxp
            ds[isp,iyp,ixm]+=wyp*wxm
            ds[isp,iyp,ixp]+=wyp*wxp
        
        ds[isp,:,:]=ds[isp,:,:]*q[isp]/ppc

    return ds

def field(ds,KX,KY,KF):
    phi=sf.fft2(np.sum(ds,axis=0))/(KX**2+KY**2+1e-30)
    ex=np.real(sf.ifft2(-1j*KX*phi))#*KF))
    ey=np.real(sf.ifft2(-1j*KY*phi))#*KF))
    
    return ex,ey

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
