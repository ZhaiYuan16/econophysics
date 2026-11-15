#!/usr/bin/env python3
"""Project 6: postprocess saved cluster, return and SOC samples only."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('results/powerlaw_refit_project6')
ROOT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261114)
B = 1000
G = 199
rows = []
plots = []

def discrete_fit(counts, lo, hi):
    k = np.arange(lo, hi+1)
    c = counts[lo:hi+1]
    n = c.sum()
    if n < 50: return None
    logk = np.log(k)
    target = c @ logk / n
    def score(a):
        w = np.exp(-a*logk); return w @ logk / w.sum()-target
    a = brentq(score, .01, 8)
    w = np.exp(-a*logk); prob = w/w.sum()
    ks = np.max(np.abs(np.cumsum(c)/n-np.cumsum(prob)))
    return a, ks, prob

def discrete_boot(group_counts, lo, hi, fit, path=False):
    group_counts=group_counts[:, :hi+1]
    nblock = len(group_counts)
    support=np.arange(lo,hi+1)
    bn=group_counts[:,lo:hi+1].sum(axis=1)
    bs=group_counts[:,lo:hi+1] @ np.log(support)
    ksims = np.empty(B)
    for b in range(B):
        ix = RNG.integers(nblock, size=nblock)
        target=bs[ix].sum()/bn[ix].sum()
        logk=np.log(support)
        def score(a):
            w=np.exp(-a*logk);return w @ logk/w.sum()-target
        ksims[b]=brentq(score,.01,8)
    obs = fit[1]
    n = int(group_counts.sum(axis=0)[lo:hi+1].sum())
    exceed = 0
    for _ in range(G):
        sim = np.zeros(group_counts.shape[1],dtype=np.int64)
        sim[lo:hi+1] = RNG.multinomial(n,fit[2])
        f = discrete_fit(sim,lo,hi)
        exceed += f[1] >= obs
    return np.nanpercentile(ksims,[2.5,97.5]),(exceed+1)/(G+1)

def cont_fit(x,lo,hi):
    v=x[(x>=lo)&(x<=hi)]
    n=len(v)
    if n<100 or hi/lo<1.2:return None
    t=np.log(hi/lo); mean=np.log(v/lo).mean()
    a=cont_alpha(mean,t)
    q=a-1
    def cdf(z):
        zz=np.clip(np.log(z/lo),0,t)
        if abs(q)<1e-6:return zz/t
        return -np.expm1(-q*zz)/-np.expm1(-q*t)
    s=np.sort(v); F=cdf(s); i=np.arange(n)
    ks=max(np.max(np.abs(F-i/n)),np.max(np.abs(F-(i+1)/n)))
    return a,ks,cdf

def cont_alpha(mean,t):
    def score(q):
        if abs(q)<1e-6:return t/2-mean
        return 1/q-t/np.expm1(q*t)-mean
    try:q=brentq(score,-30,30)
    except ValueError:return np.nan
    return q+1

def choose_cont(x):
    x=x[x>0]
    if len(x)<1000:return None
    hi=float(x.max())
    candidates=np.unique(np.quantile(x,[.70,.80,.85,.90,.925,.95]))
    fitrows=[]
    for lo in candidates:
        n=(x>=lo).sum()
        if n<500 or hi/lo<10:continue
        f=cont_fit(x,float(lo),hi)
        if f:fitrows.append((float(lo),hi,f))
    if not fitrows:return None
    return min(fitrows,key=lambda r:r[2][1])

def cont_boot(blocks,lo,hi,fit):
    # Whole configurations for lattice; whole network realizations for network data.
    nb=len(blocks); exps=np.empty(B)
    ns=np.array([((v>=lo)&(v<=hi)).sum() for v in blocks])
    ss=np.array([np.log(v[(v>=lo)&(v<=hi)]/lo).sum() for v in blocks])
    t=np.log(hi/lo)
    for b in range(B):
        ix=RNG.integers(nb,size=nb)
        exps[b]=cont_alpha(ss[ix].sum()/ns[ix].sum(),t)
    n=ns.sum()
    q=fit[0]-1; exceed=0
    for _ in range(G):
        u=RNG.random(n)
        if abs(q)<1e-6: sim=lo*np.exp(u*t)
        else: sim=lo*np.exp(-np.log1p(-u*(-np.expm1(-q*t)))/q)
        f=cont_fit(sim,lo,hi)
        exceed+=f[1]>=fit[1]
    return np.nanpercentile(exps,[2.5,97.5]),(exceed+1)/(G+1)

def record(kind,label,x,lo,hi,fit,ci,p,blocks,reason=''):
    rows.append(dict(kind=kind,label=label,n_raw=len(x),blocks=blocks,xmin=lo,xmax=hi,
        tail_n=int(((x>=lo)&(x<=hi)).sum()),alpha_pdf=fit[0],ci_low=ci[0],ci_high=ci[1],
        ks=fit[1],gof_p=p,gof_replicates=G,status='fit',note=reason))
    plots.append((kind,label,x,lo,hi,fit[0],p))

# Critical finite clusters, one largest cluster excluded per configuration.
src='results/criticality_revision_20261113/full/tables/configuration_cluster_counts.csv'
pieces=[]
for chunk in pd.read_csv(src,chunksize=500000):
    q=chunk[(chunk.p_ratio==1)&(chunk.exclusion=='largest_excluded')]
    if len(q):pieces.append(q[['length','configuration_id','cluster_size','count']])
df=pd.concat(pieces,ignore_index=True)
for L in [32,64,128,256,512]:
    d=df[df.length==L]
    ids=np.sort(d.configuration_id.unique()); maxs=int(d.cluster_size.max())
    c=np.zeros((len(ids),maxs+1),dtype=np.int32)
    c[np.searchsorted(ids,d.configuration_id),d.cluster_size]=d['count']
    pooled=c.sum(axis=0)
    # Shared rule: exclude microscopic sizes; use a sustained count-10 endpoint.
    # The L=512 endpoint is the already audited continuity endpoint.
    lo={32:5,64:10,128:10,256:10,512:15}[L]
    hi={32:54,64:127,128:170,256:306,512:430}[L]
    f=discrete_fit(pooled,lo,hi)
    ci,p=discrete_boot(c,lo,hi,f)
    x=np.repeat(np.flatnonzero(pooled),pooled[pooled>0])
    record('cluster',f'L={L}',x,lo,hi,f,ci,p,len(ids),'fixed-interval GOF; configuration CI')

def market_group(d,kind,label,blockcols):
    x=np.abs(d.z.to_numpy(dtype=float));x=x[np.isfinite(x)&(x>0)]
    selected=choose_cont(x)
    if selected is None:
        rows.append(dict(kind=kind,label=label,n_raw=len(x),blocks=d.groupby(blockcols).ngroups,
            status='insufficient_tail_range',note='No candidate with >=500 tail observations and 10x observed span'))
        plots.append((kind,label,x,np.nan,np.nan,np.nan,np.nan));return
    lo,hi,f=selected
    blocks=[np.abs(v.z.to_numpy(dtype=float)) for _,v in d.groupby(blockcols)]
    ci,p=cont_boot(blocks,lo,hi,f)
    record(kind,label,x,lo,hi,f,ci,p,len(blocks),'fixed-interval GOF; bounds selected on data')

d=pd.read_csv('results/tables/market_returns.csv.gz',usecols=['p_ratio','activity','replicate','configuration','z'])
for (p,a),v in d.groupby(['p_ratio','activity'],sort=True):
    market_group(v,'market',f'p/pc={p:g}, a={a:g}',['replicate','configuration'])
del d
d=pd.read_csv('results/tables/network_market_returns.csv.gz',usecols=['topology','relative_p','activity','network_replicate','z'])
for (top,p),v in d.groupby(['topology','relative_p'],sort=True):
    market_group(v,'network',f'{top.upper()} p/pc={p:g}',['network_replicate'])
del d

def soc_group(sub,ids,label,L,lo):
    hi=L*L
    x=sub['size'].to_numpy(dtype=int)
    if not label.startswith('legacy') and len(x)>=100:
        candidate=np.unique(x)
        candidate=candidate[(candidate>=lo)&(candidate<=x.max()/10)]
        if len(candidate)>35:candidate=np.unique(candidate[np.linspace(0,len(candidate)-1,35).astype(int)])
        counts=np.bincount(x,minlength=hi+1)[:hi+1]
        possibilities=[]
        for lower in candidate:
            if (x>=lower).sum()<100:continue
            ff=discrete_fit(counts,int(lower),hi)
            if ff:possibilities.append((ff[1],int(lower)))
        lo=min(possibilities)[1] if possibilities else max(lo,int(x.min()))
    if len(x)<100 or x.max()/lo<10 or (x>=lo).sum()<100:
        rows.append(dict(kind='soc',label=label,n_raw=len(x),blocks=len(ids),
            status='insufficient_tail_range',note='No 10x observed span with >=100 tail events'))
        plots.append(('soc',label,x,np.nan,np.nan,np.nan,np.nan));return
    c=np.zeros((len(ids),hi+1),dtype=np.int32)
    for j,pid in enumerate(ids):
        v=sub.loc[sub.path_id==pid,'size'].to_numpy(dtype=int)
        c[j]=np.bincount(v,minlength=hi+1)[:hi+1]
    f=discrete_fit(c.sum(axis=0),lo,hi)
    ci,p=discrete_boot(c,lo,hi,f,path=True)
    record('soc',label,x,lo,hi,f,ci,p,len(ids),
        'fixed-interval iid parametric GOF; path-level dependence not modeled')

for version in ['v1','v2']:
    summ=pd.read_csv(f'results/soc_research_{version}_full/path_summary.csv')
    for condition in ['baseline','cascade']:
        e=pd.read_csv(f'results/soc_research_{version}_full/{condition}/events.csv.gz')
        for L in [32,64,128]:
            burn=1000 if version=='v1' else {32:1200,64:3000,128:7500}[L]
            sub=e[(e.length==L)&(e.step>=burn)&(e['size']>0)]
            ids=sorted(summ[(summ.length==L)&(summ.condition==condition)].path_id)
            soc_group(sub,ids,f'{version} {condition} L={L}',L,21 if condition=='baseline' else 22)

legacy=pd.read_csv('results/downstream_audit_20261114/tables/soc_event_ledger.csv')
legacy=legacy.rename(columns={'removed_nodes':'size'})
soc_group(legacy,sorted(legacy.path_id.unique()),'legacy single-round L=32',32,21)

pd.DataFrame(rows).to_csv(ROOT/'refit_results.csv',index=False)

# Exactly three deliverable figures, each with all inspected groups.
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
for figidx in [1,2,3]:
    nrow=int(np.ceil(len(plots)/4))
    fig,axes=plt.subplots(nrow,4,figsize=(18,3.1*nrow),constrained_layout=True)
    for ax,item in zip(axes.flat,plots):
        kind,label,x,lo,hi,alpha,p=item
        x=np.asarray(x);x=x[x>0]
        if len(x)==0:continue
        vals,cnt=np.unique(x,return_counts=True)
        ccdf=np.cumsum(cnt[::-1])[::-1]/len(x)
        if figidx==1:
            ax.loglog(vals,ccdf,'.',ms=2,label='CCDF')
            if kind in ('cluster','soc'):
                ax.loglog(vals,cnt/len(x),'.',color='tab:orange',ms=1,alpha=.6,label='PDF')
                ax.text(.04,.05,f'N={len(x):,}; count at 1={cnt[0]:,}',transform=ax.transAxes)
            else:
                edges=np.geomspace(vals.min(),vals.max(),35)
                hist,_=np.histogram(x,bins=edges,density=True)
                mids=np.sqrt(edges[:-1]*edges[1:])
                ax.loglog(mids[hist>0],hist[hist>0],'-',color='tab:orange',lw=.8,label='binned PDF')
                ax.text(.04,.05,f'N={len(x):,}',transform=ax.transAxes)
            ax.set_ylabel('P(X≥x)')
        elif figidx==2:
            ax.loglog(vals,ccdf,'.',ms=2,color='0.35')
            if np.isfinite(lo):
                ax.axvspan(max(vals.min(),lo/100),lo,color='tab:orange',alpha=.16)
                ax.axvspan(hi,vals.max(),color='tab:red',alpha=.16)
                ax.axvspan(lo,hi,color='tab:green',alpha=.08)
                ax.axvline(lo,color='tab:green',lw=.8);ax.axvline(hi,color='tab:red',lw=.8)
                ax.text(.04,.05,f'[{lo:.3g}, {hi:.3g}]',transform=ax.transAxes)
            else:ax.text(.04,.05,'insufficient span',transform=ax.transAxes)
            ax.set_ylabel('CCDF')
        else:
            ax.loglog(vals,ccdf,'.',ms=2,label='empirical CCDF')
            if np.isfinite(lo):
                grid=np.geomspace(lo,hi,100)
                if kind in ('cluster','soc'):
                    support=np.arange(int(lo),int(hi)+1)
                    weight=support.astype(float)**(-alpha);weight/=weight.sum()
                    curve=np.cumsum(weight[::-1])[::-1]
                    ax.loglog(support,ccdf[np.searchsorted(vals,lo)]*curve,'-',lw=1,label='MLE CCDF')
                else:
                    q=alpha-1;t=np.log(hi/lo)
                    curve=np.expm1(-q*np.log(grid/hi))/np.expm1(q*t)
                    ax.loglog(grid,ccdf[np.searchsorted(vals,lo)]*curve,'-',lw=1,label='MLE CCDF')
                ax.text(.04,.05,f'α={alpha:.3f}; p={p:.3f}',transform=ax.transAxes)
            ax.set_ylabel('CCDF')
        ax.set_title(label,fontsize=8);ax.set_xlabel('|z| or size')
    for ax in axes.flat[len(plots):]:ax.axis('off')
    fig.savefig(ROOT/f'figure{figidx}.png',dpi=180)
    plt.close(fig)
print(pd.DataFrame(rows)[['kind','label','status','alpha_pdf','ci_low','ci_high','gof_p']].to_string(index=False))
