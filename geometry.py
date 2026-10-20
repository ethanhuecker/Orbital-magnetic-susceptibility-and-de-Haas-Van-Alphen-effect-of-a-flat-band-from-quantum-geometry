"""Independent analytic derivatives and contour quadrature for arXiv:2610.12193v1.

X is the paper's coefficient in epsilon=-1+B*M+B**2*X/2.
The legacy flatqwz.c2_flat returns X/2, not X.
All pocket integrals are ordinary unsigned area/line integrals.
"""
import numpy as np
from scipy.special import roots_legendre
from scipy.optimize import brentq
import flatqwz as fq

TAU = 2*np.pi

def n_derivatives(kx, ky, u=-1.2):
    kx, ky = np.broadcast_arrays(kx, ky)
    z = np.zeros_like(kx)
    sx, sy, cx, cy = np.sin(kx), np.sin(ky), np.cos(kx), np.cos(ky)
    d = np.stack((sx, sy, u+cx+cy), axis=-1)
    dx = np.stack((cx,z,-sx), axis=-1)
    dy = np.stack((z,cy,-sy), axis=-1)
    dxx = np.stack((-sx,z,-cx), axis=-1)
    dyy = np.stack((z,-sy,-cy), axis=-1)
    r = np.linalg.norm(d, axis=-1)[...,None]
    dot = lambda a,b: np.sum(a*b,axis=-1)[...,None]
    a, b = dot(d,dx), dot(d,dy)
    nx, ny = dx/r-d*a/r**3, dy/r-d*b/r**3
    nxx = dxx/r-(2*dx*a+d*(dot(dx,dx)+dot(d,dxx)))/r**3+3*d*a*a/r**5
    nyy = dyy/r-(2*dy*b+d*(dot(dy,dy)+dot(d,dyy)))/r**3+3*d*b*b/r**5
    nxy = -(dx*b+dy*a+d*dot(dx,dy))/r**3+3*d*a*b/r**5
    return d/r,nx,ny,nxx,nyy,nxy

def X_closed(kx,ky,u=-1.2):
    n,nx,ny,nxx,nyy,nxy = n_derivatives(kx,ky,u)
    U2=np.sum(nxx*nyy-nxy*nxy,axis=-1)
    return U2/4-fq.m_orb(kx,ky,u)**2

def moment_gradient(kx,ky,u=-1.2):
    sx,sy,cx,cy=np.sin(kx),np.sin(ky),np.cos(kx),np.cos(ky)
    D=sx*sx+sy*sy+(u+cx+cy)**2
    f=cx+cy+u*cx*cy
    fx,fy=-sx*(1+u*cy),-sy*(1+u*cx)
    Dx,Dy=-2*sx*(u+cy),-2*sy*(u+cx)
    return -.5*fx/D**1.5+.75*f*Dx/D**2.5, -.5*fy/D**1.5+.75*f*Dy/D**2.5

def g(kx,ky):
    return 2-np.cos(kx)-np.cos(ky)

def contour(B=1.,W=0.,g0=3.5,m0=0.,hole=True,ntheta=1024,nrad=80,u=-1.2):
    """Polar quadrature of a single pocket. No pixel counting or fitted constants.

    hole: centered at M, h=W(g-g0)+B(M-m0)>0 inside.
    electron: centered at Gamma, h<0 inside.
    dl/|grad h| = r/|partial_r h| dtheta on either contour.
    """
    th=TAU*(np.arange(ntheta)+.5)/ntheta
    ct,st=np.cos(th),np.sin(th)
    origin=np.pi if hole else 0.
    sign=1 if hole else -1
    def h(r):
        x,y=origin+r*ct,origin+r*st
        return sign*(W*(g(x,y)-g0)+B*(fq.m_orb(x,y,u)-m0))
    lo=np.zeros(ntheta)
    hi=np.pi/np.maximum(np.abs(ct),np.abs(st))
    if not (np.all(h(lo)>0) and np.all(h(hi)<0)):
        raise ValueError('Contour does not form the assumed isolated radial pocket')
    for _ in range(52):
        mid=(lo+hi)/2
        inside=h(mid)>0
        lo=np.where(inside,mid,lo);hi=np.where(inside,hi,mid)
    r=(lo+hi)/2
    x,y=origin+r*ct,origin+r*st
    mx,my=moment_gradient(x,y,u)
    hx,hy=W*np.sin(x)+B*mx,W*np.sin(y)+B*my
    radial=hx*ct+hy*st
    co=r/np.abs(radial)
    dt=TAU/ntheta
    area=.5*np.sum(r*r)*dt
    coarea=np.sum(co)*dt
    m=fq.m_orb(x,y,u)
    area_derivative=sign*np.sum((m-m0)*co)*dt
    # Berry flux, integrated inside each radial segment.
    t,w=roots_legendre(nrad); t=(t+1)/2;w=w/2
    rr=r[:,None]*t
    px=origin+rr*ct[:,None];py=origin+rr*st[:,None]
    flux=np.sum(fq.m_orb(px,py,u)*rr*r[:,None]*w)*dt
    gamma1=-flux/TAU
    # h is the energy symbol; factor B restores units of the moment contour.
    gamma2=-sign*B*np.sum(.5*X_closed(x,y,u)*co)*dt/TAU
    return dict(area=float(area),flux=float(flux),coarea=float(coarea),
                gamma1=float(gamma1),gamma2=float(gamma2),
                gamma=float(.5+gamma1+gamma2),
                dA_dB=float(area_derivative),
                Fobs=float((area-B*area_derivative)/TAU),
                Lambda_g=float(B*np.sum((g0-g(x,y))*co)*dt),
                radius=r,theta=th,kx=x,ky=y)

def scalars(d):
    return {k:v for k,v in d.items() if np.ndim(v)==0}

def global_geometry(N=512,u=-1.2):
    # Chunk rows to keep memory bounded even at N=2048.
    k=TAU*(np.arange(N)+.5)/N
    om_sum=om2_sum=u2_sum=0.;xi2=0.
    for start in range(0,N,32):
        x,y=np.meshgrid(k[start:start+32],k,indexing='ij')
        om=fq.m_orb(x,y,u)
        n,nx,ny,nxx,nyy,nxy=n_derivatives(x,y,u)
        a=np.sum(nx*nx,axis=-1);b=np.sum(ny*ny,axis=-1);c=np.sum(nx*ny,axis=-1)
        xi2=max(xi2,float(np.max((a+b+np.sqrt((a-b)**2+4*c*c))/2)))
        om_sum+=om.sum();om2_sum+=np.sum(om*om)
        u2_sum+=np.sum(nxx*nyy-nxy*nxy)
    return dict(N=N,chern=float(om_sum/N**2*TAU),I=float(om2_sum/N**2),
                U2_mean=float(u2_sum/N**2),xi2_grid=xi2,
                xi2_Gamma=1/(u+2)**2,
                moment_min=-1/(2*(u+2)**2),moment_max=1/(2*(2-u)**2))

def crossover_theory(W=2.79e-4,ntheta=1024):
    Bs=np.geomspace(2*np.pi/24000,2*np.pi/70,201)
    rows=[scalars(contour(B=B,W=W,ntheta=ntheta)) for B in Bs]
    zero=scalars(contour(B=0.,W=W,ntheta=ntheta))
    flat=scalars(contour(ntheta=ntheta))
    mid=(zero['area']+flat['area'])/(2*TAU)
    Bmid=brentq(lambda B:contour(B=B,W=W,ntheta=ntheta)['Fobs']-mid,Bs[0],Bs[-1])
    Bgamma=np.sqrt(W*flat['Lambda_g']/(TAU*abs(flat['gamma']-.5)))
    out={k:np.array([r[k] for r in rows]) for k in rows[0]}
    out.update(B=Bs,q=TAU/Bs,F_obs=out['Fobs']/TAU,
               F_nat=zero['area']/TAU**2,F_sat=flat['area']/TAU**2,
               Bmid=Bmid,Bgamma=Bgamma,Lambda_g_hole=flat['Lambda_g'],W=W)
    return out
