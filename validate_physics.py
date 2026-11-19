"""Independent radial finite-volume and algebra checks; no measured validation."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve_banded
from twin_core import thermal, predict, draw_asset, ACTIONS
ROOT=Path(__file__).resolve().parents[1]

def finite_volume(n,omega):
    h=1/n;centers=(np.arange(n)+.5)*h;left=np.arange(n)*h;right=left+h
    volumes=(right**2-left**2)/2
    gl=left/h;gr=right/h;gr[-1]=2/h
    band=np.zeros((3,n),dtype=complex)
    band[1]=gl+gr+1j*omega*volumes
    band[0,1:]=-gr[:-1];band[2,:-1]=-gl[1:]
    temp=solve_banded((1,1),band,volumes)
    mean=2*np.sum(temp*volumes)
    conduction=gr[-1]*temp[-1];storage=1j*omega*np.sum(temp*volumes)
    energy=abs(conduction+storage-.5)/.5
    return 8*mean,float(energy)

def main():
    omegas=np.logspace(-3,2,30);results={};arrays={'dimensionless_omega':omegas}
    exact=thermal(omegas/(2*np.pi),1e6,modes=80)[0]
    arrays['eigenmode_real']=exact.real;arrays['eigenmode_imag']=exact.imag
    for n in [32,64,128]:
        pairs=[finite_volume(n,w) for w in omegas];fd=np.array([p[0] for p in pairs])
        results[str(n)]={'max_absolute_complex_error':float(np.max(np.abs(fd-exact))),
                         'max_relative_energy_residual':max(p[1] for p in pairs)}
        arrays[f'fv_{n}_real']=fd.real;arrays[f'fv_{n}_imag']=fd.imag
    derivative_errors=[]
    for seed in range(90000,90005):
        t,_=draw_asset(seed);y,j=predict(t,ACTIONS)
        for k in range(6):
            step=1e-5 if k>=4 else 1e-6
            tp=t.copy();tm=t.copy();tp[k]+=step;tm[k]-=step
            fd=(predict(tp,ACTIONS)[0]-predict(tm,ACTIONS)[0])/(2*step)
            derivative_errors.append(float(np.max(np.abs(fd-j[:,k])/np.maximum(1,np.abs(j[:,k])))))
    t,s=draw_asset(99821);t2=t.copy();t2[0]-=np.log(1.10)
    first=predict(t,ACTIONS,modes=60,strength=s)[0]
    second=predict(t2,ACTIONS,modes=60,strength=s*1.10)[0]
    alias=float(np.max(np.abs(first-second)))
    trunc=float(np.max(np.abs(thermal(omegas/(2*np.pi),1e6,20)[0]-exact)))
    dc=float(abs(thermal(0.,30.,80)[0]-1))
    assert results['128']['max_absolute_complex_error']<results['64']['max_absolute_complex_error']/3
    assert max(v['max_relative_energy_residual'] for v in results.values())<1e-10
    assert max(derivative_errors)<2e-7 and alias<1e-10 and dc<1e-6
    report=dict(passed=True,finite_volume=results,max_scaled_derivative_error=max(derivative_errors),
                exact_strength_concentration_alias_max_difference=alias,
                max_20_vs_80_mode_difference=trunc,DC_normalization_error=dc,
                scope='Independent numerical checks of declared equations; no full optical mode, gas-filling or laboratory validation.')
    (ROOT/'data/physics_validation.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(ROOT/'data/physics_validation_arrays.npz',**arrays)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
