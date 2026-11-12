# Next stages

## 2. Simulator — design in progress

- Start with a simple, known demand mechanism.
- Use real data only to set plausible prices, volumes and noise scales.
- Include product substitution and store/customer differences.
- Later add a separate dynamic scenario for inventory or purchases brought forward.
- Avoid using the fitted historical model as the only ground truth.

No simulator implementation or results are included yet.

## 3. Controlled price experiments — planned

- Fixed prices as a baseline.
- Randomized variation within explicit bounds.
- Compare experiment duration, range and sample size.
- Estimate demand with Direct and separated price-response models.
- Evaluate response recovery against the known mechanism.

## 4. Policy experiments — planned

- Fit demand models to data collected during price experiments.
- Turn each model into a pricing rule by choosing the objective and constraints.
- Compare model choices and policy settings on development simulations.
- Inspect price paths, boundary hits and profit trade-offs.
- Freeze the selected rule before the separate evaluation stage.

## 5. Policy evaluation — planned

- Optimize estimated gross profit, then freeze the policy.
- Compare on new simulation periods with fixed prices and a known-mechanism oracle.
- Use repeated simulation seeds and matched random streams where appropriate.
- Report profit during data collection as well as after learning.
- Choose randomization units that respect substitution and any carry-over effects.

## 6. Contextual bandits — planned

- Compare a contextual bandit, including Thompson sampling, with fixed exploration followed by exploitation.
- Track cumulative profit, regret, learning speed and constraint violations.
- Start without intertemporal state effects; treat dynamic inventory/stockpiling as a separate extension.

The aim is to understand the methodology and trade-offs. Success in one simulator is not a real-world policy validation.
