# HC-ARF-Photothermal-Spectroscopy
A physics-informed digital twin enables uncertainty-guided measurement selection for hollow-core photothermal gas spectroscopy. Simulations demonstrate a 16.25% reduction in concentration estimation error versus fixed scanning, with calibrated uncertainty bands. Independent model tests identify robustness limitations and support reliable reporting.

## Scientific contribution and limits

The declared physics is absorption, radial heat diffusion in an equivalent cylinder and area-average optical phase. The bounded numerical controller estimates six parameters and selects from phase and paired transmission measurements using local concentration information per cost. It does not use a language model, reinforcement-learning training or real instrument commands. 

Independent complete calibration trajectories set a maximum-error quantile for each policy. This provides a finite-horizon simultaneous band under exchangeability of whole trajectories, including data-dependent stopping. It is a marginal statement over calibration and new trajectories, not a guarantee conditional on concentration, a realized calibration set, a selected report subset or distribution shift.

At matched terminal cost, the goal agent has 0.261 ppm RMSE versus 0.312 ppm for the fixed scan on 240 paired primary assets. Primary all-path coverage is 238/240, and 114/240 assets obtain a precision-qualified report. Mean acquisition used, including full budget for nonreports, is 745 ms versus 768 ms for the fixed scan. Acquisition times are declared design assumptions, not measured hardware timing.

Mismatch results must accompany these positive findings. A neighboring absorber leaves terminal goal-policy coverage at only 7/96; a 10% line-strength shift covers 64/96 and raises no verifier flag. In the changing-concentration challenge the goal policy is worse than the fixed scan. An exact concentration/strength equivalence shows why collecting the same observables cannot determine an unknown absolute absorption scale.
