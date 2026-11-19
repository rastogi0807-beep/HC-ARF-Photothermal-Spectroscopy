"""Original exact schematics and charts from executed synthetic observations."""
from pathlib import Path
from io import BytesIO
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
from twin_core import thermal, predict, draw_asset, Action, ACTIONS

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures'
POLICIES=['Goal agent','Fixed scan','Global information','Phase only']
SHORT=['Goal agent','Fixed scan','Global info.','Phase only']
COLORS=['#14877b','#3569aa','#d38a29','#a84b64']
INK='#203246';MUTED='#66798e';PALE='#eef4f7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#9aa9b5',
 'xtick.color':INK,'ytick.color':INK,'legend.frameon':False,'svg.fonttype':'none',
 'savefig.facecolor':'white','axes.titlesize':10,'axes.labelsize':9})

def save(fig,name):
    for ext in ['pdf','svg','png']:
        b=BytesIO();fig.savefig(b,format=ext,dpi=300,bbox_inches='tight',pad_inches=.10)
        (OUT/f'{name}.{ext}').write_bytes(b.getvalue())
    plt.close(fig)

def box(ax,xy,w,h,text,edge=INK,fill=PALE,size=9):
    x,y=xy;ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=.06',
                                     fc=fill,ec=edge,lw=1.1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)

def arrow(ax,a,b,color=MUTED,**kwargs):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=color,lw=1.2,
                shrinkA=3,shrinkB=3,**kwargs))

def architecture():
    fig,ax=plt.subplots(figsize=(9.2,5.2));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
    ax.text(5,5.85,'Optical measurements and bounded decision modules',ha='center',fontweight='bold',fontsize=12)
    box(ax,(.2,4.5),1.65,.65,'Tunable pump\nAM at 1–36 kHz',edge=COLORS[2])
    box(ax,(.2,3.55),1.65,.65,'1550 nm probe\nPhase reference',edge=COLORS[1])
    box(ax,(2.35,4.0),.7,.7,'WDM',edge=COLORS[1])
    arrow(ax,(1.85,4.8),(2.35,4.5),COLORS[2]);arrow(ax,(1.85,3.85),(2.35,4.2),COLORS[1])
    ax.add_patch(FancyBboxPatch((3.55,4.05),3.1,.6,boxstyle='round,pad=.03,rounding_size=.15',fc='#e0f0ee',ec=COLORS[0],lw=1.4))
    ax.plot([3.6,6.6],[4.35,4.35],color=COLORS[2],lw=2.5)
    ax.plot([3.6,6.6],[4.22,4.22],color=COLORS[1],lw=1.6)
    arrow(ax,(3.05,4.35),(3.55,4.35));arrow(ax,(6.65,4.35),(7.2,4.35))
    ax.text(5.1,4.95,'Gas-filled HC-ARF   •   equilibrated sample',ha='center',fontsize=9)
    box(ax,(7.2,4.0),.7,.7,'WDM',edge=COLORS[1])
    box(ax,(8.35,4.85),1.4,.65,'Pump detector\nOn/off ratio',edge=COLORS[2])
    box(ax,(8.35,3.65),1.4,.65,'Interferometer\nComplex phase',edge=COLORS[1])
    arrow(ax,(7.9,4.5),(8.35,5.1),COLORS[2]);arrow(ax,(7.9,4.2),(8.35,3.95),COLORS[1])
    # Cross-section is a schematic, not a computed optical field.
    cx,cy=1.1,2.1
    ax.add_patch(Circle((cx,cy),.75,fc='#f3f6f8',ec=MUTED,lw=1))
    for angle in np.arange(7)*2*np.pi/7:
        ax.add_patch(Circle((cx+.46*np.cos(angle),cy+.46*np.sin(angle)),.17,fc='white',ec=COLORS[1],lw=1))
    ax.add_patch(Circle((cx,cy),.25,fc='#d0ebe5',ec=COLORS[0]))
    ax.text(cx,1.05,'Schematic cross-section',ha='center',fontsize=8)
    box(ax,(2.55,2.05),1.95,.7,'Experiment planner\nInformation per cost',edge=COLORS[0],size=7.8)
    box(ax,(5.1,2.05),1.95,.7,'Uncertainty verifier\nPath band + innovations',edge=COLORS[2],size=7.3)
    box(ax,(7.65,2.05),1.95,.7,'Twin estimator\nConcentration + nuisances',edge=COLORS[1],size=7.3)
    arrow(ax,(8.85,3.6),(8.6,2.8));arrow(ax,(7.65,2.4),(7.05,2.4));arrow(ax,(5.1,2.4),(4.5,2.4))
    box(ax,(4.4,.55),3.5,.65,'Supervisor\nMeasure / report / withhold',edge=INK,size=8.5)
    arrow(ax,(3.55,2.05),(4.7,1.25));arrow(ax,(8.6,2.05),(7.5,1.25))
    arrow(ax,(3.55,2.8),(3.55,3.55),COLORS[0]);ax.text(3.65,3.25,'Detuning, frequency, modality',va='center',fontsize=8)
    ax.text(5,.08,'Deterministic numerical agents; simulated observations only.',ha='center',fontsize=8,color=MUTED)
    save(fig,'figure_1_architecture')

def flowchart():
    fig,ax=plt.subplots(figsize=(8.6,4.7));ax.set(xlim=(0,10),ylim=(.5,7));ax.axis('off')
    box(ax,(.2,5.95),2.6,.85,'Offline calibration\n128 independent full paths\nFreeze 95% path quantile',edge=COLORS[1],size=8)
    box(ax,(3.6,6.05),3.,.7,'Acquire initial design\nIndependent current records',edge=COLORS[0],size=8)
    box(ax,(3.6,4.9),3.,.7,'Fit twin and covariance\nCheck residuals and innovations',edge=COLORS[0],size=7.7)
    arrow(ax,(5.1,6.05),(5.1,5.6))
    ax.plot([2.85,3.2,3.2],[6.35,6.35,5.55],color=MUTED,lw=1.1)
    arrow(ax,(3.2,5.55),(3.6,5.55))
    diamond=np.array([[5.1,4.55],[6.4,3.95],[5.1,3.35],[3.8,3.95]])
    ax.add_patch(Polygon(diamond,fc='#fff7e6',ec=COLORS[2],lw=1.2))
    ax.text(5.1,3.95,'Unflagged and\nprecision target met?',ha='center',va='center',fontsize=7.8)
    arrow(ax,(5.1,4.9),(5.1,4.55))
    box(ax,(7.2,3.6),2.5,.7,'Report concentration\nand calibrated band',edge=COLORS[0],size=8)
    arrow(ax,(6.4,3.95),(7.2,3.95),COLORS[0]);ax.text(6.65,4.1,'Yes',fontsize=8,color=COLORS[0])
    diamond2=np.array([[5.1,3.0],[6.3,2.5],[5.1,2.0],[3.9,2.5]])
    ax.add_patch(Polygon(diamond2,fc='#fff7e6',ec=COLORS[2],lw=1.2))
    ax.text(5.1,2.5,'Budget\nremaining?',ha='center',va='center',fontsize=8)
    arrow(ax,(5.1,3.35),(5.1,3.0));ax.text(5.22,3.15,'No',fontsize=7.5)
    box(ax,(7.2,2.15),2.5,.7,'Withhold report\nPrecision target not achieved',edge=COLORS[3],size=7.5)
    arrow(ax,(6.3,2.5),(7.2,2.5),COLORS[3]);ax.text(6.65,2.65,'No',fontsize=8,color=COLORS[3])
    box(ax,(.2,2.05),2.7,.9,'Score actions per unit cost\nAcquire next measurement\nUpdate acquisition budget',edge=COLORS[1],size=7.7)
    arrow(ax,(3.9,2.5),(2.9,2.5),COLORS[1]);ax.text(3.25,2.65,'Yes',fontsize=8,color=COLORS[1])
    ax.plot([1.55,1.55,3.25],[3.0,5.25,5.25],color=MUTED,lw=1.1)
    arrow(ax,(3.25,5.25),(3.6,5.25))
    ax.text(5,.95,'Complete paths are retained for matched-budget evaluation.\nThe online policy never uses true concentration.',ha='center',fontsize=8,color=MUTED)
    save(fig,'figure_2_flowchart')

def forward_figures():
    fig,axs=plt.subplots(1,3,figsize=(10,3.1),layout='constrained')
    f=np.logspace(2,5,180)
    for t,col in zip([18,30,65],COLORS):
        h,_=thermal(f,t);axs[0].semilogx(f/1000,np.abs(h),label=f'{t} μs',color=col)
        axs[1].semilogx(f/1000,np.angle(h,deg=True),color=col)
    axs[0].set(xlabel='Modulation frequency (kHz)',ylabel='Normalized thermal amplitude',title='(a) Cylindrical diffusion twin');axs[0].legend(fontsize=8)
    axs[1].set(xlabel='Modulation frequency (kHz)',ylabel='Thermal phase (degrees)',title='(b) Informative phase roll-off')
    data=np.load(ROOT/'data/physics_validation_arrays.npz')
    for n,col in zip([32,64,128],COLORS):
        d=np.abs((data[f'fv_{n}_real']+1j*data[f'fv_{n}_imag'])-(data['eigenmode_real']+1j*data['eigenmode_imag']))
        axs[2].loglog(data['dimensionless_omega'],d,label=f'{n} cells',color=col)
    axs[2].set(xlabel='Dimensionless angular frequency',ylabel='Absolute complex error',title='(c) Independent finite volumes');axs[2].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.15)
    save(fig,'figure_3_physics')

def primary_figures(summary,frame):
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.2),layout='constrained')
    ss=[summary['primary'][p] for p in POLICIES]
    vals=np.array([s['RMSE_ppm'] for s in ss]);cis=np.array([s['RMSE_CI'] for s in ss]).T
    axs[0].bar(np.arange(4),vals,color=COLORS,width=.62,yerr=[vals-cis[0],cis[1]-vals],capsize=4)
    axs[0].set(xticks=np.arange(4),xticklabels=SHORT,ylabel='Terminal concentration RMSE (ppm)',title='(a) Equal cost: 24 units per asset')
    axs[0].tick_params(axis='x',labelsize=8)
    for i,v in enumerate(vals):axs[0].text(i,cis[1,i]+.03,f'{v:.3f}',ha='center',fontsize=8)
    axs[0].set_ylim(0,1.60)
    goal=frame[frame.policy=='Goal agent']
    fixed=frame[frame.policy=='Fixed scan']
    axs[1].scatter(goal.true_c,goal.error,s=12,alpha=.6,color=COLORS[0],label='Goal agent')
    axs[1].scatter(fixed.true_c,fixed.error,s=12,alpha=.45,color=COLORS[1],marker='x',label='Fixed scan')
    axs[1].axhline(0,lw=.8,color=MUTED);axs[1].set(xlabel='True concentration (ppm)',ylabel='Terminal estimate minus truth (ppm)',title='(b) All 240 paired primary assets');axs[1].legend(fontsize=8)
    save(fig,'figure_4_estimation')

def trajectory():
    episodes=json.loads((ROOT/'data/primary_episodes.json').read_text());cal=json.loads((ROOT/'data/calibration_thresholds.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(9.3,3.15),layout='constrained')
    for ax,p,col in zip(axs,POLICIES[:2],COLORS[:2]):
        e=next(e for e in episodes if e['seed']==51000 and e['policy']==p)
        s=e['states'];x=np.array([v['spent'] for v in s])*40
        estimate=np.exp([v['theta'][0] for v in s]);sd=np.array([v['sd_log_c'] for v in s]);q=cal[p]['q_path']
        ax.fill_between(x,estimate*np.exp(-q*sd),estimate*np.exp(q*sd),color=col,alpha=.18,label='Calibrated path band')
        ax.plot(x,estimate,'o-',markersize=3,color=col,label='Twin estimate');ax.axhline(np.exp(e['truth'][0]),color=INK,ls='--',label='True concentration')
        ax.set(xlabel='Assumed acquisition time (ms)',ylabel='Concentration (ppm)',title=p+'   •   first primary asset')
        ax.legend(fontsize=7.5,loc='best')
    save(fig,'figure_5_trajectory')

def coverage(summary,frame):
    fig,axs=plt.subplots(1,2,figsize=(9.3,3.25),layout='constrained')
    levels=np.array([.8,.9,.95,.975,.99])
    for p,col in zip(POLICIES,COLORS):
        g=frame[frame.policy==p]
        rates=[np.mean(g.max_z<=summary['calibration'][p]['levels'][str(float(v))]) for v in levels]
        axs[0].plot(levels*100,np.array(rates)*100,'o-',label=p,color=col,markersize=3)
    axs[0].plot([80,100],[80,100],'--',color=MUTED,lw=1);axs[0].set(xlabel='Nominal simultaneous coverage (%)',ylabel='Observed all-path coverage (%)',xlim=(79,100),ylim=(76,101),title='(a) Primary trajectories, n = 240 per policy');axs[0].legend(fontsize=7.5)
    x=np.arange(4);w=.24
    for shift,key,label,color in [(-w,'path_raw_covered','Uncalibrated 1.96 SD','#d38a29'),(0,'path_terminal_cal_covered','Terminal-only calibration','#8aa4bd'),(w,'path_covered','Whole-path calibration','#14877b')]:
        y=[summary['primary'][p][key]['rate']*100 for p in POLICIES]
        axs[1].bar(x+shift,y,width=w,label=label,color=color)
    axs[1].axhline(95,color=MUTED,ls='--',lw=.8);axs[1].set(xticks=x,xticklabels=SHORT,ylabel='Observed all-path coverage (%)',ylim=(0,105),title='(b) Coverage across every update');axs[1].tick_params(axis='x',labelsize=8);axs[1].legend(fontsize=7.5,loc='upper center',bbox_to_anchor=(.5,-.18))
    save(fig,'figure_6_coverage')

def allocation():
    episodes=json.loads((ROOT/'data/primary_episodes.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(9.8,3.1),gridspec_kw={'width_ratios':[1,1,.8]},layout='constrained')
    matrices=[]
    for p in ['Goal agent','Global information']:
        counts=np.zeros((4,5))
        for e in episodes:
            if e['policy']==p:
                for r in e['records'][6:]:
                    if r['mode']=='P':counts[[1000,4000,12000,36000].index(r['frequency']),[-3,-1,0,1,3].index(r['x'])]+=1
        matrices.append(counts/240)
    vmax=max(m.max() for m in matrices)
    for ax,mat,p in zip(axs[:2],matrices,['Goal agent','Global information']):
        im=ax.imshow(mat,origin='lower',aspect='auto',cmap='YlGnBu',vmin=0,vmax=vmax)
        for i in range(4):
            for j in range(5):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if mat[i,j]>vmax*.55 else INK)
        ax.set(xticks=range(5),xticklabels=[-3,-1,0,1,3],yticks=range(4),yticklabels=[1,4,12,36],xlabel='Normalized detuning',ylabel='Frequency (kHz)',title=p)
    fig.colorbar(im,ax=list(axs[:2]),label='Additional phase records per asset',shrink=.75,pad=.02)
    means=[sum(r['mode']=='D' for e in episodes if e['policy']==p for r in e['records'])/240 for p in POLICIES]
    axs[2].barh(SHORT,means,color=COLORS);axs[2].invert_yaxis();axs[2].set(xlabel='Transmission pairs\nper full-budget asset',title='Anchor allocation');axs[2].tick_params(axis='y',labelsize=8)
    save(fig,'figure_7_allocation')

def stopping(summary,frame):
    fig,axs=plt.subplots(1,2,figsize=(9.3,3.15),layout='constrained')
    for p,col in zip(POLICIES,COLORS):
        g=frame[frame.policy==p];cost=np.arange(0,25)
        fraction=[np.sum(g.report&(g.stop_cost<=b))/len(g)*100 for b in cost]
        axs[0].step(cost*40,fraction,where='post',label=p,color=col)
    axs[0].set(xlabel='Assumed acquisition time (ms)',ylabel='Assets with a precision-qualified report (%)',ylim=(0,65),xlim=(0,980),title='(a) All-case reporting incidence');axs[0].legend(fontsize=8)
    labels=['1–3 ppm','3–8 ppm','8–20 ppm'];x=np.arange(3);w=.2
    for i,(p,col) in enumerate(zip(POLICIES,COLORS)):
        d=summary['concentration_bins'][p]
        vals=[d[name.replace('–','-')]['mean_stop_ms'] for name in labels]
        axs[1].bar(x+(i-1.5)*w,vals,width=w,color=col,label=p)
    axs[1].set(xticks=x,xticklabels=labels,ylabel='Mean acquisition used (ms)',ylim=(0,1030),title='(b) Nonreports consume the full budget');axs[1].legend(fontsize=7.5,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2)
    save(fig,'figure_8_stopping')

def challenges(summary):
    settings=['Double noise','Extra thermal pole','Spectral interferent','Line strength +10%','Concentration changes','Heavy tails']
    fig,axs=plt.subplots(1,2,figsize=(9.1,4),layout='constrained')
    for ax,key,title,cmap in zip(axs,['terminal_covered','final_flag'],['(a) Terminal bands covering truth / 96','(b) Full-path verifier flags / 96'],['Blues','Oranges']):
        arr=np.array([[summary['challenges'][s][p][key]['count'] for p in POLICIES[:2]] for s in settings])
        ax.imshow(arr,vmin=0,vmax=96,aspect='auto',cmap=cmap)
        for i in range(len(settings)):
            for j in range(2):ax.text(j,i,str(arr[i,j]),ha='center',va='center',fontsize=10,color='white' if arr[i,j]>65 else INK)
        ax.set(xticks=[0,1],xticklabels=POLICIES[:2],yticks=range(6),yticklabels=settings,title=title)
        if ax is axs[1]:ax.set_yticklabels([])
    save(fig,'figure_9_challenges')

def equivalence():
    theta=np.array([np.log(10.),0.,np.log(30.),.1,0.,0.]);other=theta.copy();other[0]-=np.log(1.10)
    xs=np.linspace(-4,4,160);aa=[Action('P',float(x),4000.) for x in xs];dd=[Action('D',float(x)) for x in xs]
    a=predict(theta,aa,modes=60,strength=1.)[0][::2];b=predict(other,aa,modes=60,strength=1.10)[0][::2]
    c=predict(theta,dd,modes=60,strength=1.)[0];d=predict(other,dd,modes=60,strength=1.10)[0]
    fig,axs=plt.subplots(1,2,figsize=(9.3,3.05),layout='constrained')
    for ax,y,z,title,unit in [(axs[0],a,b,'(a) Photothermal in-phase signal','Phase (μrad)'),(axs[1],c*1e6,d*1e6,'(b) Transmission log-ratio','Log-ratio (×10⁻⁶)')]:
        ax.plot(xs,y,color=COLORS[0],lw=2,label='10 ppm, line scale 1.00')
        ax.plot(xs,z,color=COLORS[2],ls='--',lw=1.5,label='9.091 ppm, line scale 1.10')
        ax.set(xlabel='Normalized detuning',ylabel=unit,title=title);ax.legend(fontsize=8)
    save(fig,'figure_10_equivalence')

def graphical_abstract(summary):
    fig,ax=plt.subplots(figsize=(8.2,3.15));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    ax.text(5,3.85,'Agentic physics-informed gas sensing',ha='center',fontweight='bold',fontsize=13)
    box(ax,(.25,2.25),2.3,.9,'HC-ARF measurements\nPhase quadratures\nTransmission pairs',edge=COLORS[1],size=8.5)
    box(ax,(3.2,2.25),3.2,.9,'Heat-diffusion twin\nSelect next measurement\nCalibrate full trajectories',edge=COLORS[0],size=8.5)
    box(ax,(7.05,2.25),2.65,.9,'Concentration + band\nReport or keep measuring',edge=COLORS[2],size=8.5)
    arrow(ax,(2.55,2.7),(3.2,2.7));arrow(ax,(6.4,2.7),(7.05,2.7))
    ax.text(2.4,1.35,'240 synthetic primary assets\n0.261 vs 0.312 ppm terminal RMSE\nEqual acquisition budget',ha='center',va='center',fontsize=9.3,color=COLORS[0])
    ax.text(7.4,1.35,'238 / 240 whole paths covered\n114 / 240 precision reports\nNominal 95% path calibration',ha='center',va='center',fontsize=9.3,color=COLORS[1])
    ax.text(5,.25,'Limits: line-strength bias is unobservable; an unmodelled absorber breaks nominal coverage.',ha='center',fontsize=9,color=COLORS[3])
    save(fig,'Graphical_Abstract')

def main():
    OUT.mkdir(exist_ok=True)
    summary=json.loads((ROOT/'data/results_summary.json').read_text())
    frame=pd.read_csv(ROOT/'data/episode_metrics.csv');primary=frame[frame.setting=='Nominal']
    architecture();flowchart();forward_figures();primary_figures(summary,primary)
    trajectory();coverage(summary,primary);allocation();stopping(summary,primary)
    challenges(summary);equivalence();graphical_abstract(summary)
    print('Generated 10 manuscript figures and a graphical abstract in PDF, SVG and 300-dpi PNG.')

if __name__=='__main__':main()
