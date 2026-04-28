**Contextual Thompson Sampling with Time-Dependent Effects**
This repository implements and compares several Thompson Sampling bandit algorithms, including standard contextual bandits and time-aware extensions. The main focus is understanding how time since last intervention affects decision making and learning 

**Repository Structure**
heartsteps_action_centered_ts.py
  Main implementation on real-world data. Uses contextual features from suggestions.csv and includes time-aware decision making based on user history.

thompson_sampling_centered_reward.py
  Baseline contextual Thompson Sampling model with centered reward without time dependency. Trained on synthetic data.

thompson_sampling_centered_reward_with_time_dep.py
  Synthetic environment with explicit time-dependent reward structure. Used to test whether adding temporal features improves learning.

suggestions.csv
Real-world dataset containing user interactions, context features, timestamps, and observed outcomes.
Data from Klasnja, P., Smith, S., Seewald, N. J., Lee, A., Hall, K., Luers, B., Hekler, E. B., & Murphy, S. A. (2019). Efficacy of Contextually Tailored Suggestions for Physical Activity: A Micro-randomized Optimization Trial of HeartSteps. Annals of Behavioral Medicine, 53(6), 573–582. https://doi.org/10.1093/abm/kay067

error_regret_overt_time_not_time_dep.png
  Learning curve and regret plot for the baseline model.
error_regret_overt_time_with_time_dep.png
  Learning curve and regret plot for the time-dependent model.

**Core Idea**
We study a contextual bandit problem where the reward depends on both context and time:

r_t(a) = f(s_{t,a}) + g(Δt) + noise

where Δt represents the time since the last action. The goal is to learn both contextual effects and temporal effects jointly.

Key Insight

Standard contextual bandits ignore time between actions. This repository attempts to show that incorporating time improves:

regret reduction
intervention efficiency

**Algorithm Overview**
At each timestep:

Observe context for all actions
Sample parameter vector using Thompson Sampling
Score each action using sampled model
Select best action (or no action)
Convert score into probability
Sample final action stochastically
Observe reward
Update posterior distribution over parameters
Model Update Rule

The model uses a Bayesian linear regression style update:

B ← B + π_t (1 - π_t) s s^T
b ← b + s (indicator - π_t) r
θ_hat = B^{-1} b

**Running the Code**

Baseline model:
python thompson_sampling_centered_reward.py

Time-dependent synthetic model:
python thompson_sampling_centered_reward_with_time_dep.py

Real-world model:
python heartsteps_action_centered_ts.py
