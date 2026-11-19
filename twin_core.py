"""Grey-box cylindrical heat twin and bounded sequential experiment controller.

All records in this project are simulated. Phase is in microradians, gas
concentration in ppm, and diffusion time in microseconds in parameter storage.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.special import jn_zeros
from scipy.stats import chi2

LENGTH = 1.0
ALPHA = 3e-5  # m^-1 ppm^-1, declared effective line-design value, not HITRAN
PUMP = .05
KAPPA = .026
DNDT = -9e-7
LAMBDA_PROBE = 1.55e-6
PHASE_SCALE = abs(DNDT) * PUMP / (4 * LAMBDA_PROBE * KAPPA) * 1e6
SIGMA_PHASE = 1.5
SIGMA_ABS = 1.5e-5
OFF = 8.0
BUDGET = 24
TARGET_LOG = np.log(1.20)
MEAN = np.array([np.log(5.), 0., np.log(30.), 0., 0., 0.])
PRIOR_SD = np.array([1.2, .20, .35, .5, 12., 12.])
LOW = np.array([np.log(.1), np.log(.4), np.log(5.), -2., -80., -80.])
HIGH = np.array([np.log(100.), np.log(2.5), np.log(150.), 2., 80., 80.])
J2 = jn_zeros(0, 80)**2
J2_TWIN = J2[:20]
J2_GENERATOR = J2[:60]
FREQUENCIES = [1000., 4000., 12000., 36000.]


@dataclass(frozen=True)
class Action:
    mode: str
    x: float
    frequency: float = 0.

    @property
    def cost(self) -> int:
        return 1 if self.mode == "P" else 3

    @property
    def key(self) -> str:
        return f"{self.mode}:{self.x:g}:{self.frequency:g}"


PTS = [Action("P", x, f) for x in [-3., -1., 0., 1., 3.]
       for f in FREQUENCIES]
DAS = [Action("D", x) for x in [-1., 0., 1.]]
ACTIONS = PTS + DAS
INITIAL = [Action("P", 0, 1000), Action("P", -1, 12000),
           Action("P", 1, 36000), Action("P", 3, 4000),
           Action("D", 0), Action("D", 1)]
INITIAL_PHASE = INITIAL[:4] + [Action("P", 0, 4000), Action("P", -3, 36000)]
FIXED = [Action("P", -1, 1000), Action("P", 0, 4000), Action("P", 1, 12000),
         Action("P", 3, 36000), Action("D", 0), Action("D", -1)]


def thermal(f, t_us, modes=20):
    """Uniform heat / area-average probe, Dirichlet cylindrical wall, exp(+iwt)."""
    z = J2[:modes]
    w = np.asarray(f)[..., None] * (2j*np.pi*t_us*1e-6)
    den = z + w
    h = np.sum(32/(z*den), axis=-1)
    dh_logtime = np.sum(-32*w/(z*den**2), axis=-1)
    return h, dh_logtime


def lorentz(x, delta):
    v = np.asarray(x)-delta
    return 1/(1+v*v), 2*v/(1+v*v)**2


def predict(theta, actions, modes=20, strength=1., second_line=0., gas_scale=1.,
            thermal_extra=0.):
    """Return flattened real observations and their analytic parameter Jacobian."""
    c, gain, t_us = np.exp(theta[:3]); delta, br, bi = theta[3:]
    out = []; jac = []
    for a in actions:
        ell, dl = lorentz(a.x, delta)
        extra, _ = lorentz(a.x-1.5, -.0)
        tau = ALPHA*LENGTH*c*gas_scale*strength*ell
        if a.mode == "P":
            h, dh = thermal(a.frequency, t_us, modes)
            if thermal_extra:
                h2, _ = thermal(a.frequency, t_us*8, modes)
                h = (1-thermal_extra)*h + thermal_extra*h2
                # Generator-only option; returned Jacobian is not used.
            absorbed = -np.expm1(-tau)
            response = PHASE_SCALE*gain*absorbed*h
            hw = 1/(1+2j*np.pi*a.frequency*500e-6)
            y = response+(br+1j*bi)*hw
            if second_line:
                y += PHASE_SCALE*gain*ALPHA*LENGTH*second_line*extra*h
            v = np.zeros(6, dtype=complex)
            v[0] = PHASE_SCALE*gain*tau*np.exp(-tau)*h
            v[1] = response
            v[2] = PHASE_SCALE*gain*absorbed*dh
            v[3] = PHASE_SCALE*gain*ALPHA*LENGTH*c*gas_scale*strength*dl*np.exp(-tau)*h
            v[4] = hw; v[5] = 1j*hw
            out.extend([y.real, y.imag]); jac.extend([v.real, v.imag])
        else:
            eo, do = lorentz(OFF, delta)
            y = ALPHA*LENGTH*c*gas_scale*strength*(ell-eo)
            if second_line:
                exo, _ = lorentz(OFF-1.5, 0.)
                y += ALPHA*LENGTH*second_line*(extra-exo)
            v = np.zeros(6)
            v[0] = y; v[3] = ALPHA*LENGTH*c*gas_scale*strength*(dl-do)
            out.append(y); jac.append(v)
    return np.asarray(out), np.asarray(jac)


def noise_scales(actions):
    return np.array([v for a in actions for v in
                    ([SIGMA_PHASE]*2 if a.mode == "P" else [SIGMA_ABS])])


def fit(actions, data, start):
    sig = noise_scales(actions)
    cache = {}

    def calc(t):
        if not cache or not np.array_equal(t, cache['t']):
            y, j = predict(t, actions)
            cache.update(t=t.copy(), r=np.r_[(y-data)/sig, (t-MEAN)/PRIOR_SD],
                         j=np.vstack([j/sig[:, None], np.diag(1/PRIOR_SD)]))
        return cache['r'], cache['j']

    opt = least_squares(lambda t: calc(t)[0], np.clip(start, LOW+1e-9, HIGH-1e-9),
                        jac=lambda t: calc(t)[1], bounds=(LOW, HIGH),
                        xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=80,
                        x_scale=PRIOR_SD)
    res, j = calc(opt.x)
    precision = j.T@j
    cov = np.linalg.inv(precision)
    return dict(theta=opt.x, cov=cov, rss=float(np.sum(res[:-6]**2)),
                success=bool(opt.success), nfev=int(opt.nfev))


def information(theta, cov, action, objective):
    _, j = predict(theta, [action]); j = j/noise_scales([action])[:, None]
    v = cov@j.T
    middle = np.eye(len(j))+j@v
    after = cov-v@np.linalg.solve(middle, v.T)
    if objective == "goal":
        return .5*np.log(cov[0, 0]/max(after[0, 0], 1e-14))/action.cost
    return .5*np.linalg.slogdet(middle)[1]/action.cost


def choose(policy, theta, cov, spent, index):
    allowed = [a for a in (PTS if policy == "Phase only" else ACTIONS)
               if a.cost <= BUDGET-spent]
    if policy == "Fixed scan":
        proposed = FIXED[index % len(FIXED)]
        return proposed if proposed.cost <= BUDGET-spent else PTS[index % len(PTS)]
    objective = "global" if policy == "Global information" else "goal"
    values = [information(theta, cov, a, objective) for a in allowed]
    return allowed[int(np.argmax(values))]


def draw_asset(seed):
    rng = np.random.default_rng(seed)
    c = np.exp(rng.uniform(np.log(1.), np.log(20.)))
    gain = np.clip(np.exp(rng.normal(0, .15)), .65, 1.5)
    td = np.clip(np.exp(rng.normal(np.log(30.), .20)), 18., 65.)
    delta = rng.uniform(-.45, .45)
    b = rng.normal(0, 6., 2)
    strength = np.exp(rng.normal(0, .02))
    theta = np.array([np.log(c), np.log(gain), np.log(td), delta, *b])
    return theta, float(strength)


def simulate(seed, policy, setting="Nominal"):
    began = time.perf_counter()
    truth, strength = draw_asset(seed)
    observations = []; actions = []; records = []; states = []
    counts = {}; spent = 0; failures = []; theta = MEAN.copy(); cov = np.diag(PRIOR_SD**2)
    initial = INITIAL_PHASE if policy == "Phase only" else INITIAL
    index = 0
    while spent < BUDGET:
        a = initial[index] if index < len(initial) else choose(
            policy, theta, cov, spent, index-len(initial))
        previous_theta = theta.copy(); previous_cov = cov.copy()
        count = counts.get(a.key, 0); counts[a.key] = count+1
        aid = ACTIONS.index(a)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 177, aid, count]))
        kwargs = dict(modes=60, strength=strength)
        if setting == "Extra thermal pole": kwargs['thermal_extra'] = .25
        if setting == "Spectral interferent": kwargs['second_line'] = 3.
        if setting == "Line strength +10%": kwargs['strength'] *= 1.10
        if setting == "Concentration changes" and spent >= 14: kwargs['gas_scale'] = 1.40
        y, _ = predict(truth, [a], **kwargs)
        sigma = noise_scales([a])
        if setting == "Double noise": sigma = 2*sigma
        noise = rng.normal(size=len(y))
        if setting == "Heavy tails": noise = rng.standard_t(3, size=len(y))/np.sqrt(3)
        observed = y + sigma*noise
        innovation = None
        if index >= len(initial):
            pred, jj = predict(previous_theta, [a])
            vv = np.diag(noise_scales([a])**2)+jj@previous_cov@jj.T
            difference = observed-pred
            innovation = float(difference@np.linalg.solve(vv, difference))
            if innovation > chi2.ppf(1-.01/BUDGET, len(observed)):
                failures.append(index)
        actions.append(a); observations.extend(observed); spent += a.cost
        current_c = float(np.exp(truth[0])*kwargs.get('gas_scale', 1.))
        records.append(dict(index=index, action=a.key, mode=a.mode, x=a.x,
                            frequency=a.frequency, cost=a.cost, spent=spent,
                            expected=y.tolist(), observed=observed.tolist(),
                            sigma_true=sigma.tolist(), innovation=innovation,
                            true_c=current_c))
        if index >= len(initial)-1:
            fitted = fit(actions, np.array(observations), theta)
            theta, cov = fitted['theta'], fitted['cov']
            if not fitted['success']: failures.append(index)
            if fitted['rss'] > chi2.ppf(.999, len(observations)):
                failures.append(index)
            states.append(dict(index=index, spent=spent, theta=theta.tolist(),
                               cov=cov.tolist(), rss=fitted['rss'], nfev=fitted['nfev'],
                               success=fitted['success'], flagged=bool(failures),
                               sd_log_c=float(np.sqrt(cov[0, 0])),
                               true_c=current_c,
                               log_error=float(abs(theta[0]-np.log(current_c)))))
        index += 1
    return dict(seed=int(seed), policy=policy, setting=setting,
                truth=truth.tolist(), line_strength=strength,
                records=records, states=states, elapsed=time.perf_counter()-began)
