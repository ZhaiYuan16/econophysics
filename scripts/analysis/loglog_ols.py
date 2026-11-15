#!/usr/bin/env python3
"""Descriptive log-log CCDF interval OLS on saved simulation observations."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'results/loglog_ols_comparison'
OUT.mkdir(parents=True,exist_ok=True)
ROWS=[];PANELS={k:[] for k in ['cluster','market','network','soc','degree']}
MIN_POINTS=14
MIN_DECADES=1.0
MIN_INTERVAL_OBS=100
MIN_ABOVE_UPPER=20

def empirical(values=None,counts=None):
    if counts is not None:
        x=np.flatnonzero(counts>0).astype(float)
        x=x[x>0]
        n=counts[x.astype(int)].astype(float)
    else:
        v=np.asarray(values,dtype=float)
        v=v[np.isfinite(v)&(v>0)]
        x,n=np.unique(v,return_counts=True)
        n=n.astype(float)
    if len(x)==0:return None
    remaining=np.cumsum(n[::-1])[::-1]
    return x,remaining/remaining[0],remaining

def interval_fit(x,ccdf,remaining):
    # Sample existing points approximately evenly in log(x), preserving the
    # original empirical CCDF for plotting. No MLE/KS/Bootstrap is used.
    if len(x)<MIN_POINTS:return None,'too_few_distinct_points'
    low=x[0]
    # The upper tail begins at the 70th percentile for returns; for integer
    # cluster/event/degree data the caller has already supplied a lower floor.
    grid=np.geomspace(low,x[-1],100)
    ix=np.unique(np.clip(np.searchsorted(x,grid),0,len(x)-1))
    xx=x[ix];yy=ccdf[ix];rr=remaining[ix]
    good=(rr>=MIN_ABOVE_UPPER)&(yy>0)&(xx>0)
    xx=xx[good];yy=yy[good];rr=rr[good]
    if len(xx)<MIN_POINTS:return None,'too_few_tail_points'
    lx=np.log10(xx);ly=np.log10(yy)
    candidates=[]
    for i in range(len(lx)-MIN_POINTS+1):
        for j in range(i+MIN_POINTS-1,len(lx)):
            if lx[j]-lx[i]<MIN_DECADES:continue
            if rr[i]-rr[j]<MIN_INTERVAL_OBS:continue
            X=lx[i:j+1];Y=ly[i:j+1]
            b,a=np.polyfit(X,Y,1)
            fitted=a+b*X
            sst=np.sum((Y-Y.mean())**2)
            r2=1-np.sum((Y-fitted)**2)/sst if sst>0 else -np.inf
            candidates.append((r2,lx[j]-lx[i],i,j,a,b))
    if not candidates:return None,'no_interval_meets_width_and_count'
    best_r2=max(t[0] for t in candidates)
    # R² leads; among nearly equal fits choose the wider interval.
    eligible=[t for t in candidates if t[0]>=best_r2-.002]
    r2,width,i,j,a,b=max(eligible,key=lambda t:(t[1],t[0]))
    mid=(i+j)//2
    b_left=np.polyfit(lx[i:mid+1],ly[i:mid+1],1)[0]
    b_right=np.polyfit(lx[mid:j+1],ly[mid:j+1],1)[0]
    drift=abs(b_left-b_right)/max(abs(b),.01)
    status='visible_linear_candidate' if r2>=.98 and b<0 and drift<=.25 else 'no_clear_linear_scaling'
    return dict(xmin=xx[i],xmax=xx[j],n_points=j-i+1,n_interval=int(rr[i]-rr[j]),
                slope=b,intercept=a,r2=r2,ccdf_alpha=-b,pdf_tau=1-b,
                span_decades=width,slope_half_relative_difference=drift,status=status),None

def add(kind,label,values=None,counts=None,floor=None,source='',note=''):
    e=empirical(values,counts)
    if e is None:
        ROWS.append(dict(kind=kind,label=label,status='no_positive_data',source=source,note=note));return
    x,y,remaining=e
    # Returns: inspect the upper 30% of nonzero absolute returns. Integer
    # quantities: include the observed distribution from the stated floor.
    if kind in ('market','network'):
        target=.30*remaining[0]
        lower=np.flatnonzero(remaining<=target)
        xfloor=x[lower[0]] if len(lower) else x[-1]
    else:xfloor=max(x[0],floor or 1)
    keep=x>=xfloor
    fit,reason=interval_fit(x[keep],y[keep],remaining[keep])
    row=dict(kind=kind,label=label,n_raw=int(remaining[0]),source=source,note=note,
             scanned_from=xfloor,curve_points=len(x))
    if fit:row.update(fit)
    else:row.update(status=reason)
    ROWS.append(row)
    PANELS[kind].append((label,x,y,fit,xfloor))

def main():
    source='results/criticality_revision_20261113/full/tables/configuration_cluster_counts.csv'
    pooled={L:None for L in [32,64,128,256,512]}
    for chunk in pd.read_csv(BASE/source,chunksize=500000):
        q=chunk[(chunk.p_ratio==1)&(chunk.exclusion=='largest_excluded')]
        for L,g in q.groupby('length'):
            c=np.bincount(g.cluster_size,weights=g['count'],minlength=L*L+1)
            pooled[L]=c if pooled[L] is None else pooled[L]+c
    for L,c in pooled.items():
        add('cluster',f'L={L}',counts=c,floor=2,source=source,
            note='finite clusters; largest per configuration excluded')

    source='results/tables/market_returns.csv.gz'
    d=pd.read_csv(BASE/source,usecols=['p_ratio','activity','z'])
    for (p,a),g in d.groupby(['p_ratio','activity']):
        add('market',f'p/pc={p:g}, a={a:g}',values=np.abs(g.z.to_numpy()),source=source)
    del d
    source='results/tables/network_market_returns.csv.gz'
    d=pd.read_csv(BASE/source,usecols=['topology','relative_p','z'])
    for (top,p),g in d.groupby(['topology','relative_p']):
        add('network',f'{top.upper()} p/pc={p:g}',values=np.abs(g.z.to_numpy()),source=source)
    del d

    for ver in ['v1','v2']:
        for condition in ['baseline','cascade']:
            source=f'results/soc_research_{ver}_full/{condition}/events.csv.gz'
            d=pd.read_csv(BASE/source)
            burns={32:1000,64:1000,128:1000} if ver=='v1' else {32:1200,64:3000,128:7500}
            for L in [32,64,128]:
                v=d.loc[(d.length==L)&(d.step>=burns[L])&(d['size']>0),'size'].to_numpy(dtype=int)
                add('soc',f'{ver} {condition} L={L}',counts=np.bincount(v,minlength=L*L+1),
                    floor=2,source=source,note='burn-in removed')
    source='results/downstream_audit_20261114/tables/soc_event_ledger.csv'
    d=pd.read_csv(BASE/source)
    add('soc','legacy single-round L=32',counts=np.bincount(d.removed_nodes.to_numpy(dtype=int)),
        floor=2,source=source)

    source='results/tables/network_degree_distributions.csv'
    d=pd.read_csv(BASE/source)
    for top,g in d.groupby('topology'):
        c=np.bincount(g.degree,weights=g.node_count)
        add('degree',top.upper(),counts=c,floor=1,source=source,
            note='exact degree multiplicities; Lattice constant degree 4' if top=='lattice' else 'exact degree multiplicities')

    summary=pd.DataFrame(ROWS)
    summary.insert(0,'model',summary.kind.map({'cluster':'critical_percolation',
        'market':'CB_market','network':'CB_network_market','soc':'SOC_dissipation',
        'degree':'network_degree'}))
    summary.insert(2,'condition',summary.label)
    summary.insert(3,'fitted_distribution','CCDF')
    summary.to_csv(OUT/'ols_summary.csv',index=False)
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    for kind,panels in PANELS.items():
        cols=3;nr=int(np.ceil(len(panels)/cols))
        fig,axes=plt.subplots(nr,cols,figsize=(15,4*nr),constrained_layout=True)
        for ax,(label,x,y,fit,xfloor) in zip(np.ravel(axes),panels):
            if len(x)>3000:
                ix=np.unique(np.linspace(0,len(x)-1,3000).astype(int));xp=x[ix];yp=y[ix]
            else:xp=x;yp=y
            ax.loglog(xp,yp,'.',ms=2,color='tab:blue',label='empirical CCDF')
            if fit:
                grid=np.geomspace(fit['xmin'],fit['xmax'],80)
                line=10**fit['intercept']*grid**fit['slope']
                ax.loglog(grid,line,'-',color='tab:red',lw=1.5,label='interval OLS')
                ax.axvline(fit['xmin'],color='tab:green',lw=.8,ls='--')
                ax.axvline(fit['xmax'],color='tab:green',lw=.8,ls='--')
                note=f"[{fit['xmin']:.3g}, {fit['xmax']:.3g}]\nb={fit['slope']:.3f}; α={fit['ccdf_alpha']:.3f}; τ={fit['pdf_tau']:.3f}\nR²={fit['r2']:.4f}; {fit['status']}"
            else:note='no qualifying interval'
            ax.text(.04,.04,note,transform=ax.transAxes,fontsize=7)
            ax.set_title(label,fontsize=8)
            ax.set_ylabel('empirical CCDF P(X ≥ x)')
            ax.set_xlabel({'cluster':'cluster size s','market':'absolute standardized return |z|',
                'network':'absolute standardized return |z|','soc':'removed nodes S','degree':'node degree k'}[kind])
        for ax in np.ravel(axes)[len(panels):]:ax.axis('off')
        np.ravel(axes)[0].legend(loc='upper right',fontsize=7)
        fig.savefig(OUT/f'{kind}_loglog_ols.png',dpi=170);plt.close(fig)
    print(summary[['kind','label','status','xmin','xmax','slope','pdf_tau','r2']].to_string(index=False))

def refined_scan(x,y,remaining,kind):
    """Find sustained CCDF declines; retain distinct candidates for review."""
    grid=np.geomspace(x[0],x[-1],145)
    ix=np.unique(np.clip(np.searchsorted(x,grid),0,len(x)-1))
    xx=x[ix]; yy=y[ix]; rr=remaining[ix]
    upper_count=10 if kind=='network' else 5
    good=(rr>=upper_count)&(yy>0)
    xx=xx[good];yy=yy[good];rr=rr[good]
    if len(xx)<7:return [],'too_few_curve_points'
    lx=np.log10(xx);ly=np.log10(yy)
    all_rows=[]
    for i in range(len(xx)-6):
        if kind=='network' and yy[i]>.98:continue  # reject only the near-unity saturation platform
        for j in range(i+6,len(xx)):
            span=lx[j]-lx[i]
            if span<(.65 if kind=='network' else .5):continue
            nobs=rr[i]-rr[j]
            if nobs<(100 if kind=='network' else 25):continue
            drop=ly[i]-ly[j]
            if drop<(.15 if kind=='network' else .12):continue
            X=lx[i:j+1];Y=ly[i:j+1]
            slope,intercept=np.polyfit(X,Y,1)
            if slope>=0:continue
            pred=intercept+slope*X
            sst=np.sum((Y-Y.mean())**2)
            r2=1-np.sum((Y-pred)**2)/sst if sst>0 else -np.inf
            one=i+(j-i)//3;two=i+2*(j-i)//3
            pieces=[(i,one),(one,two),(two,j)]
            slopes=[];drops=[]
            for aa,bb in pieces:
                if bb-aa<2:break
                slopes.append(np.polyfit(lx[aa:bb+1],ly[aa:bb+1],1)[0])
                drops.append(ly[aa]-ly[bb])
            if len(slopes)<3 or min(drops)<.015:continue
            drift=(max(slopes)-min(slopes))/max(abs(slope),.02)
            stable=all(s<0 for s in slopes) and drift<(.35 if kind=='network' else .7)
            if kind=='network':
                tier='standard' if span>=1 and len(X)>=14 and r2>=.98 and stable else 'local'
                if tier=='local' and (r2<.965 or drift>.85):continue
            else:
                tier='standard' if span>=1 and len(X)>=12 and nobs>=80 and r2>=.97 and stable else 'exploratory'
                if tier=='exploratory' and (span>=1 or r2<.94 or drift>1.0):continue
            all_rows.append(dict(xmin=xx[i],xmax=xx[j],i=i,j=j,n_points=len(X),
                n_interval=int(nobs),n_above_upper=int(rr[j]),start_ccdf=yy[i],end_ccdf=yy[j],
                log_drop=drop,span_decades=span,slope=slope,intercept=intercept,r2=r2,
                ccdf_alpha=-slope,pdf_tau=1-slope,third_slope_spread=drift,tier=tier))
    if not all_rows:
        # Separate a shortage of data/range from a visibly curved decline.
        available=np.log10(xx[-1]/xx[0]) if xx[-1]>xx[0] else 0
        required_span=.65 if kind=='network' else .5
        if available<required_span:return [],'insufficient_scale_range'
        minimum_obs=100 if kind=='network' else 25
        has_populated_window=any(lx[j]-lx[i]>=required_span and rr[i]-rr[j]>=minimum_obs
            for i in range(len(lx)) for j in range(i+1,len(lx)))
        if not has_populated_window:return [],'insufficient_events_in_window'
        return [],'no_stable_decline'
    all_rows.sort(key=lambda r:(r['tier']!='standard',-r['r2'],-r['span_decades']))
    # Preserve distinct intervals: alternatives must shift an endpoint
    # materially in log space, not merely move by one sampled point.
    chosen=[]
    for row in all_rows:
        if any(abs(np.log10(row['xmin']/q['xmin']))<.22 and
               abs(np.log10(row['xmax']/q['xmax']))<.22 for q in chosen):continue
        chosen.append(row)
        if len(chosen)==5:break
    return chosen,None

def refined_add(kind,label,values=None,counts=None,source='',note=''):
    e=empirical(values,counts)
    if e is None:return dict(kind=kind,label=label,status='no_positive_data',source=source,note=note),None,[]
    x,y,remaining=e
    candidates,reason=refined_scan(x,y,remaining,kind)
    final=candidates[0] if candidates else None
    row=dict(kind=kind,label=label,n_raw=int(remaining[0]),source=source,note=note,
             candidate_count=len(candidates))
    if final:
        row.update({k:v for k,v in final.items() if k not in ('i','j')})
        row['status']='standard_candidate' if final['tier']=='standard' else 'exploratory_local_only'
    else:row['status']=reason
    candidate_rows=[]
    for rank,item in enumerate(candidates,1):
        candidate_rows.append(dict(kind=kind,label=label,rank=rank,**{k:v for k,v in item.items() if k not in ('i','j')}))
    return row,(label,x,y,final,reason),candidate_rows

def refined_plot(panels,kind,path):
    cols=3;nr=int(np.ceil(len(panels)/cols))
    fig,axes=plt.subplots(nr,cols,figsize=(15,4.1*nr),constrained_layout=True)
    for ax,(label,x,y,fit,reason) in zip(np.ravel(axes),panels):
        if len(x)>3500:
            ix=np.unique(np.linspace(0,len(x)-1,3500).astype(int));xp=x[ix];yp=y[ix]
        else:xp=x;yp=y
        ax.loglog(xp,yp,'.',ms=2,color='tab:blue',label='empirical CCDF')
        if fit:
            grid=np.geomspace(fit['xmin'],fit['xmax'],80)
            ax.loglog(grid,10**fit['intercept']*grid**fit['slope'],color='tab:red',lw=1.5,label='selected OLS')
            ax.axvline(fit['xmin'],color='tab:green',ls='--',lw=.8)
            ax.axvline(fit['xmax'],color='tab:green',ls='--',lw=.8)
            desc=f"{fit['tier']}: [{fit['xmin']:.3g}, {fit['xmax']:.3g}]\nα={fit['ccdf_alpha']:.3f}, τ={fit['pdf_tau']:.3f}, R²={fit['r2']:.3f}\nspan={fit['span_decades']:.2f} decades"
        else:desc=f'no selected line: {reason}'
        ax.text(.04,.04,desc,transform=ax.transAxes,fontsize=7)
        ax.set_title(label,fontsize=8);ax.set_ylabel('CCDF P(X ≥ x)')
        ax.set_xlabel('absolute standardized return |z|' if kind=='network' else 'removed nodes S')
    for ax in np.ravel(axes)[len(panels):]:ax.axis('off')
    np.ravel(axes)[0].legend(loc='upper right',fontsize=7)
    fig.savefig(path,dpi=170);plt.close(fig)

def refine_network_soc():
    old=pd.read_csv(OUT/'ols_summary.csv')
    rows=[];candidates=[];panels={'network':[],'soc':[]}
    source='results/tables/network_market_returns.csv.gz'
    d=pd.read_csv(BASE/source,usecols=['topology','relative_p','z'])
    for (top,p),g in d.groupby(['topology','relative_p']):
        label=f'{top.upper()} p/pc={p:g}'
        row,panel,cs=refined_add('network',label,values=np.abs(g.z.to_numpy()),source=source)
        rows.append(row);panels['network'].append(panel);candidates.extend(cs)
    for ver in ['v1','v2']:
        for condition in ['baseline','cascade']:
            source=f'results/soc_research_{ver}_full/{condition}/events.csv.gz'
            d=pd.read_csv(BASE/source)
            burns={32:1000,64:1000,128:1000} if ver=='v1' else {32:1200,64:3000,128:7500}
            for L in [32,64,128]:
                v=d.loc[(d.length==L)&(d.step>=burns[L])&(d['size']>0),'size'].to_numpy(dtype=int)
                label=f'{ver} {condition} L={L}'
                row,panel,cs=refined_add('soc',label,counts=np.bincount(v,minlength=L*L+1),source=source,
                    note=f'burn_in={burns[L]}; positive event sizes')
                rows.append(row);panels['soc'].append(panel);candidates.extend(cs)
    source='results/downstream_audit_20261114/tables/soc_event_ledger.csv'
    d=pd.read_csv(BASE/source)
    row,panel,cs=refined_add('soc','legacy single-round L=32',
        counts=np.bincount(d.removed_nodes.to_numpy(dtype=int)),source=source)
    rows.append(row);panels['soc'].append(panel);candidates.extend(cs)
    fresh=pd.DataFrame(rows)
    fresh.insert(0,'model',fresh.kind.map({'network':'CB_network_market','soc':'SOC_dissipation'}))
    fresh.insert(2,'condition',fresh.label)
    fresh.insert(3,'fitted_distribution','CCDF')
    combined=pd.concat([old[~old.kind.isin(['network','soc'])],fresh],ignore_index=True)
    combined.to_csv(OUT/'ols_summary_v2.csv',index=False)
    pd.DataFrame(candidates).to_csv(OUT/'candidate_intervals_v2.csv',index=False)
    refined_plot(panels['network'],'network',OUT/'network_loglog_ols_v2.png')
    refined_plot(panels['soc'],'soc',OUT/'soc_loglog_ols_v2.png')
    print(fresh[['kind','label','status','xmin','xmax','slope','r2','candidate_count']].to_string(index=False))

def soc_short_candidates(counts):
    """Third tier for L=128: exact observed-size points, never log-binned."""
    x,ccdf,remaining=empirical(counts=counts)
    lx=np.log10(x);ly=np.log10(ccdf)
    found=[]
    for i in range(len(x)-7):
        for j in range(i+7,len(x)):
            span=lx[j]-lx[i]
            if not .25<=span<.5:continue
            if remaining[i]-remaining[j]<20 or remaining[j]<8:continue
            drop=ly[i]-ly[j]
            if drop<.15:continue
            X=lx[i:j+1];Y=ly[i:j+1]
            slope,intercept=np.polyfit(X,Y,1)
            if slope>=0:continue
            pred=intercept+slope*X
            r2=1-np.sum((Y-pred)**2)/np.sum((Y-Y.mean())**2)
            if r2<.985:continue
            thirds=[(i,i+(j-i)//3),(i+(j-i)//3,i+2*(j-i)//3),(i+2*(j-i)//3,j)]
            local=[np.polyfit(lx[a:b+1],ly[a:b+1],1)[0] for a,b in thirds]
            if any(v>=0 for v in local):continue
            drift=(max(local)-min(local))/abs(slope)
            if drift>.35:continue
            found.append(dict(xmin=x[i],xmax=x[j],n_points=j-i+1,
                n_interval=int(remaining[i]-remaining[j]),n_above_upper=int(remaining[j]),
                start_ccdf=ccdf[i],end_ccdf=ccdf[j],log_drop=drop,
                span_decades=span,slope=slope,intercept=intercept,r2=r2,
                ccdf_alpha=-slope,pdf_tau=1-slope,third_slope_spread=drift,tier='short_local'))
    if not found:return []
    peak=max(v['r2'] for v in found)
    # As before, a near-tie goes to the wider segment, then keep alternatives.
    best=max((v for v in found if v['r2']>=peak-.002),key=lambda v:(v['span_decades'],v['r2']))
    ordered=[best]+sorted((v for v in found if v is not best),key=lambda v:(-v['r2'],-v['span_decades']))
    distinct=[]
    for v in ordered:
        if any(abs(np.log10(v['xmin']/w['xmin']))<.035 and
               abs(np.log10(v['xmax']/w['xmax']))<.035 for w in distinct):continue
        distinct.append(v)
        if len(distinct)==5:break
    return distinct

def soc_v3_plot(rows,raw,path):
    fig,axes=plt.subplots(5,3,figsize=(15,20.5),constrained_layout=True)
    style={'standard_candidate':('tab:red','-','standard'),
        'exploratory_local_only':('tab:orange','--','exploratory'),
        'short_local':('tab:purple',':','short_local')}
    for ax,(_,row) in zip(axes.flat,rows.iterrows()):
        label=row.label
        x,y,_=empirical(counts=raw[label])
        ax.loglog(x,y,'.',color='tab:blue',ms=2,label='empirical CCDF')
        status=row.status
        if status in style:
            color,linestyle,tier=style[status]
            grid=np.geomspace(row.xmin,row.xmax,80)
            ax.loglog(grid,10**row.intercept*grid**row.slope,color=color,ls=linestyle,lw=1.7,label=tier+' OLS')
            ax.axvline(row.xmin,color='tab:green',ls='--',lw=.8)
            ax.axvline(row.xmax,color='tab:green',ls='--',lw=.8)
            info=f"{tier}: [{row.xmin:.3g}, {row.xmax:.3g}]\nτ={row.pdf_tau:.3f}; R²={row.r2:.3f}; width={row.span_decades:.2f}"
        else:info=f'no fit: {status}'
        ax.text(.04,.04,info,transform=ax.transAxes,fontsize=7)
        ax.set_title(label,fontsize=8);ax.set_ylabel('CCDF P(S ≥ s)');ax.set_xlabel('removed nodes S')
    for ax in axes.flat[len(rows):]:ax.axis('off')
    axes.flat[0].legend(loc='upper right',fontsize=7)
    fig.savefig(path,dpi=170);plt.close(fig)

def refine_soc_v3():
    prior=pd.read_csv(OUT/'ols_summary_v2.csv')
    rows=prior[prior.kind=='soc'].copy().reset_index(drop=True)
    earlier=pd.read_csv(OUT/'candidate_intervals_v2.csv')
    candidates=earlier[earlier.kind=='soc'].copy().reset_index(drop=True)
    candidates['candidate_origin']='v2_standard_or_exploratory'
    targets={'v1 baseline L=128','v1 cascade L=128','v2 cascade L=128'}
    review='v2 baseline L=128'
    raw={}
    for version in ['v1','v2']:
        for condition in ['baseline','cascade']:
            source=f'results/soc_research_{version}_full/{condition}/events.csv.gz'
            d=pd.read_csv(BASE/source)
            burn={32:1000,64:1000,128:1000} if version=='v1' else {32:1200,64:3000,128:7500}
            for L in [32,64,128]:
                label=f'{version} {condition} L={L}'
                values=d.loc[(d.length==L)&(d.step>=burn[L])&(d['size']>0),'size'].to_numpy(dtype=int)
                raw[label]=np.bincount(values,minlength=L*L+1)
                if label not in targets and label!=review:continue
                options=soc_short_candidates(raw[label])
                if label in targets and options:
                    at=rows.index[rows.label==label][0]
                    for key,value in options[0].items():rows.at[at,key]=value
                    rows.at[at,'status']='short_local'
                    rows.at[at,'candidate_count']=len(options)
                    rows.at[at,'note']=f'burn_in={burn[L]}; short segment only; low confidence'
                for rank,opt in enumerate(options,1):
                    candidates.loc[len(candidates)]=dict(kind='soc',label=label,rank=rank,
                        candidate_origin='v3_short_local',**opt)
    source='results/downstream_audit_20261114/tables/soc_event_ledger.csv'
    d=pd.read_csv(BASE/source)
    raw['legacy single-round L=32']=np.bincount(d.removed_nodes.to_numpy(dtype=int))
    rows.to_csv(OUT/'soc_summary_v3.csv',index=False)
    candidates.to_csv(OUT/'soc_candidate_intervals_v3.csv',index=False)
    soc_v3_plot(rows,raw,OUT/'soc_loglog_ols_v3.png')
    print(rows[['label','status','xmin','xmax','slope','r2','span_decades','n_points']].to_string(index=False))

if __name__=='__main__':refine_soc_v3()
