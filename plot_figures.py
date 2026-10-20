"""Five numbered paper figures and the alternate Figure 4 caption protocol.

Plot definitions and style adapted from the supplied make_fig*.py files,
Copyright 2026 Anthropic PBC, SPDX-License-Identifier: Apache-2.0.
All theoretical constants are loaded from freshly evaluated geometry.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import CubicSpline
import flatqwz as fq
import geometry as geo
from paper3_style import (apply_paper3_style,ed_points,panel_letter,clean_log_ticks,
                         C_THEORY,C_ED,C_FIELD,C_META,COLW,TEXTW)
from reproduce import ROOT,CONFIG
from analyze import chi_closed

def save(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.png')
    fig.savefig(ROOT/'figures'/f'{name}.svg',metadata={'Date':None})
    plt.close(fig)

def moment():
    k=np.linspace(0,2*np.pi,601);x,y=np.meshgrid(k,k,indexing='ij')
    M=fq.m_orb(x,y,CONFIG['u'])
    fig,ax=plt.subplots(figsize=(.74*COLW,2.30),constrained_layout=True)
    im=ax.pcolormesh(x,y,M,norm=TwoSlopeNorm(vmin=M.min(),vcenter=0.,vmax=M.max()),
        cmap='RdBu_r',shading='auto',rasterized=True)
    cb=fig.colorbar(im,ax=ax,fraction=.052,pad=.03);cb.set_label(r'$\mathcal{M}$',labelpad=2)
    cb.set_ticks([M.min(),-.4,0.,M.max()]);cb.set_ticklabels([f'{M.min():.2f}','-0.40','0',f'{M.max():.2f}'])
    ax.contour(x,y,M,levels=[0],colors=['.15'],linewidths=1.1,linestyles='--')
    zone_axes(ax)
    ax.annotate(r'$\mathcal{C}_0$',xy=(np.pi+1.05,np.pi+1.05),fontsize=7,color='.15')
    save(fig,'figure_1_moment')

def zone_axes(ax):
    ax.set_xlabel(r'$k_x$');ax.set_ylabel(r'$k_y$',labelpad=1)
    ax.set_xticks([0,np.pi,2*np.pi],['0',r'$\pi$',r'$2\pi$'])
    ax.set_yticks([0,np.pi,2*np.pi],['0',r'$\pi$',r'$2\pi$'])
    ax.set_xlim(0,2*np.pi);ax.set_ylim(0,2*np.pi);ax.set_aspect('equal')

def spectrum_panels(d):
    fig,axs=plt.subplots(1,2,figsize=(COLW,1.8),constrained_layout=True)
    for ax,q,y,label,letter in zip(axs,[d['bandwidth_q'],d['spacing_q']],
             [d['bandwidth'],d['spacing']],['bandwidth',r'$\hbar\tilde\omega_{\rm eff}$'],'ab'):
        B=2*np.pi/q;Bd=np.geomspace(B.min()*.8,B.max()*1.25,100)
        pred=d['moment_span']*Bd if letter=='a' else 2*np.pi*Bd**2/d['lever']
        ed_points(ax,B,y,ms=3.6,label='ED' if letter=='a' else r'ED at $\mu=\varepsilon_0$')
        ax.loglog(Bd,pred,color=C_THEORY,lw=1.2,label=r'Theory ($\sim B$)' if letter=='a' else r'Theory ($\sim B^2$)')
        ax.set_xlabel('$B$');ax.set_ylabel(label)
        ax.legend(loc='upper left',handlelength=1.2,borderaxespad=.05,handletextpad=.5)
        clean_log_ticks(ax);panel_letter(ax,letter,dx=-.30 if letter=='a' else -.36,dy=1.)
    save(fig,'figure_2_spectrum')

def susceptibility(g,chi):
    fig,ax=plt.subplots(figsize=(.9*COLW,2.35),constrained_layout=True)
    mus=np.linspace(-1.75,.3,600)
    ax.axhline(0,color='.85',lw=.6)
    ax.plot(mus,chi_closed(mus,.1,g['I']),color=C_THEORY,lw=1.2,label='closed form, no fit')
    rows=chi['mu_scan_T0.1']
    ed_points(ax,[r['mu'] for r in rows],[r['chi_ED'] for r in rows],ms=4.2,label='ED')
    ax.set_xlabel(r'$\mu$');ax.set_ylabel(r'$\chi_{\rm thermo}$')
    ax.annotate(r'$-I$',xy=(0.,-g['I']),xytext=(-.42,-.046),fontsize=7,color=C_META,
        ha='center',arrowprops=dict(arrowstyle='-',color=C_META,lw=.6,shrinkB=2.5))
    ax.legend(loc='upper right',borderaxespad=.1,handlelength=1.4);ax.margins(x=.03)
    save(fig,'figure_3_susceptibility')

def dhva(d,flat,caption=False):
    fig,(axa,axb)=plt.subplots(1,2,figsize=(COLW,1.85),width_ratios=[1.35,1.],constrained_layout=True)
    axa.plot(d['theory_q']/(2*np.pi),d['waveform_theory'],color=C_THEORY,lw=1.,label='LK formula, no fit')
    ed_points(axa,d['waveform_q']/(2*np.pi),d['waveform_ed'],ms=2.8,mew=.5,label='ED')
    axa.set_xlabel(r'$1/B$');axa.set_ylabel('$(2\\pi/B)^3\\,\\delta\\Xi_{\\rm osc}$\n(per site)')
    axa.legend(loc='upper right',handlelength=1.2,borderaxespad=.1,handletextpad=.5,fontsize=6)
    axa.set_ylim(-.123,.205);panel_letter(axa,'a',dx=-.28)
    rows=flat['damping_caption' if caption else 'damping_pinned']['rows']
    X=np.linspace(1e-6,6.4,400)
    axb.plot(X,X/np.sinh(X),color=C_THEORY,lw=1.,label=r'$X/\sinh X$')
    ed_points(axb,[r['X'] for r in rows],[r['D_meas'] for r in rows],ms=3.4,label='ED')
    axb.set_xlabel(r'$X=2\pi^2T/\hbar\tilde\omega_{\rm eff}$')
    axb.set_ylabel(r'$A_1(T)/A_1(0)$');axb.set_ylim(0,1.06);axb.set_xlim(0,6.5)
    axb.legend(loc='upper right',handlelength=1.2,borderaxespad=.1,fontsize=6)
    panel_letter(axb,'b',dx=-.3)
    save(fig,'figure_4_caption_protocol' if caption else 'figure_4_dhva')

def crossover(cross):
    d=np.load(ROOT/'data/theory_crossover.npz');B=d['B'];F=d['Fobs']
    mid=float(d['Bmid']);W=CONFIG['W'];Fsat=float(d['F_sat'])*2*np.pi;Fnat=float(d['F_nat'])*2*np.pi
    interp=CubicSpline(np.log(B),F)
    fig,(axa,axb)=plt.subplots(1,2,figsize=(.86*TEXTW,2.30),width_ratios=[.82,1.30],constrained_layout=True)
    for bb,c in zip([0,mid,5*mid],C_FIELD):
        co=geo.contour(B=bb,W=W)
        axa.plot(np.r_[co['kx'],co['kx'][0]],np.r_[co['ky'],co['ky'][0]],color=c,lw=1.2)
    co=geo.contour();axa.plot(np.r_[co['kx'],co['kx'][0]],np.r_[co['ky'],co['ky'][0]],color='.3',lw=1.,ls='--')
    zone_axes(axa);panel_letter(axa,'a',dx=-.24)
    for i,(c,lb) in enumerate(zip(C_FIELD,[r'$B\to0$',f'$B={mid:.3f}$',f'$B={5*mid:.3f}$'])):
        axa.text(.18,2*np.pi-.42-.62*i,lb,color=c,fontsize=6.5,
                 bbox=dict(facecolor='white',edgecolor='none',pad=.6,alpha=.9))
    axa.text(np.pi,np.pi+.52,r'$A_{\rm FS}(0)$',color=C_FIELD[0],fontsize=7,ha='center')
    axa.text(np.pi,np.pi-2.62,r'$\mathcal{M}=0$',color='.3',fontsize=7,ha='center')
    axa.text(np.pi,np.pi-1.55,r'$A_{\mathcal{M}}$',color='.3',fontsize=7,ha='center')
    rows=cross['frequency'];be=np.array([r['B'] for r in rows]);fe=[r['F_ED'] for r in rows]
    ed_points(axb,be,fe,ms=3.2,mew=.6,label='ED')
    bg=np.geomspace(be.min()/1.25,be.max()*1.12,300)
    axb.semilogx(bg,interp(np.log(bg)),lw=1.2,color=C_THEORY,label='composite contour')
    bd=np.geomspace(.65*mid,be.max()*1.12,80)
    axb.plot(bd,Fsat-2*W*float(d['Lambda_g_hole'])/(2*np.pi*bd),ls=':',lw=1.6,color='.25',label=r'$1/B$ law')
    axb.axhline(Fnat,ls='--',lw=.8,color=C_FIELD[0],alpha=.45)
    axb.axhline(Fsat,ls='--',lw=.8,color='.3',alpha=.45)
    axb.text(be.min()/1.15,Fnat+.075,r'$A_{\rm FS}(0)/2\pi$',color=C_FIELD[0],fontsize=7)
    axb.text(be.min()/1.15,Fsat-.19,r'$A_{\mathcal{M}}/2\pi$',color='.3',fontsize=7)
    for bb,c in zip([mid,5*mid],C_FIELD[1:]):
        axb.plot([bb],[interp(np.log(bb))],'o',ms=5,mfc=c,mec='white',mew=.7,zorder=6)
    axb.plot([be.min()],[Fnat],'o',ms=5,mfc=C_FIELD[0],mec='white',mew=.7,zorder=6)
    axb.set_xlabel('$B$');axb.set_ylabel(r'$F_{\rm obs}$',labelpad=2);axb.set_ylim(.30,2.55)
    axb.legend(loc='center left',bbox_to_anchor=(.02,.60),handlelength=1.4)
    panel_letter(axb,'b',dx=-.14);save(fig,'figure_5_crossover')

def main():
    apply_paper3_style();plt.rcParams['svg.hashsalt']='2610.12193v1'
    g=json.loads((ROOT/'data/geometry.json').read_text())
    flat=json.loads((ROOT/'data/flat_analysis.json').read_text())
    chi=json.loads((ROOT/'data/susceptibility.json').read_text())
    cross=json.loads((ROOT/'data/crossover_analysis.json').read_text())
    d=np.load(ROOT/'data/flat_observables.npz')
    moment();spectrum_panels(d);susceptibility(g,chi)
    dhva(d,flat);dhva(d,flat,caption=True);crossover(cross)
    print('Wrote five numbered paper figures and the Figure 4 caption-protocol comparison.',flush=True)

if __name__=='__main__':main()
