import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# ============================================================
# ACTION-CENTERED THOMPSON SAMPLING BANDIT
# ============================================================

class ActionCenteredThompson:
    """
    Contextual bandit with:
    - Thompson sampling (uncertainty-driven exploration)
    - Binary decision: 0 (no action) vs 1..N (choose best arm)
    - Time-dependent features (recency effects)

    Core idea:
    Learn theta such that:
        reward ≈ theta^T * (context + time features)
    """

    def __init__(self, d, N, pi_min=0.2, pi_max=0.8, v=1.0):

        # feature dimension + 2 time features
        self.d = d + 2
        self.N = N

        # probability clipping for stability
        self.pi_min = pi_min
        self.pi_max = pi_max

        # exploration strength (higher = more random sampling)
        self.v = v

        # Bayesian linear regression state
        self.B = np.eye(self.d)        # precision matrix (confidence)
        self.b = np.zeros(self.d)      # reward-weighted features
        self.theta_hat = np.zeros(self.d)

        # tracks last time a non-zero action was taken
        self.last_action_time = -1

    # ----------------------------------------------------------

    def sample_theta(self):
        """
        Sample a plausible model from uncertainty.

        If model is uncertain:
        → covariance is large
        → more exploration
        """
        cov = self.v ** 2 * np.linalg.inv(self.B)
        return np.random.multivariate_normal(self.theta_hat, cov)

    # ----------------------------------------------------------

    def choose_action(self, context_features, t):
        """
        Decide which action to take at time t.

        Steps:
        1. Add time-based features (recency)
        2. Sample model (Thompson sampling)
        3. Score all actions
        4. Pick best candidate action (a_bar)
        5. Convert score → probability π_t
        6. Sample final action (0 or a_bar)
        """

        # time since last intervention
        if self.last_action_time == -1:
            delta_t = t
        else:
            delta_t = t - self.last_action_time

        delta_squared = delta_t ** 2

        # augment each arm with time features
        augmented_features = []
        for s in context_features:

            # extend context with:
            # [bias-like term, time decay signal]
            s_aug = np.concatenate([s, [1.0, delta_squared]])
            augmented_features.append(s_aug)

        # sample model from posterior
        theta_prime = self.sample_theta()

        # score each arm under sampled model
        scores = [s @ theta_prime for s in augmented_features]

        # pick best arm under sampled model
        a_bar = int(np.argmax(scores))
        s_bar = augmented_features[a_bar]

        # predict mean reward under current estimate
        mean = s_bar @ self.theta_hat

        # uncertainty (variance of prediction)
        var = s_bar @ np.linalg.inv(self.B) @ s_bar
        std = np.sqrt(self.v ** 2 * var)

        # probability reward > 0 (Gaussian assumption)
        p_positive = 1 - norm.cdf(0, loc=mean, scale=std)

        # clip probability for stability
        pi_t = np.clip(p_positive, self.pi_min, self.pi_max)

        # final stochastic decision
        if np.random.rand() < pi_t:
            action = a_bar + 1   # convert index → action space (1..N)
        else:
            action = 0           # no intervention

        return action, a_bar, pi_t, s_bar, delta_t

    # ----------------------------------------------------------

    def update(self, s_bar, action, pi_t, reward, t):
        """
        Update model using observed reward.

        Key idea:
        - Only update if action was actually taken
        - Use weighted Bayesian regression update
        """

        indicator = 1 if action > 0 else 0

        # confidence update (more certainty after observations)
        self.B += pi_t * (1 - pi_t) * np.outer(s_bar, s_bar)

        # reward update
        self.b += s_bar * (indicator - pi_t) * reward

        # solve linear system for new theta
        self.theta_hat = np.linalg.inv(self.B) @ self.b

        # update recency tracker
        if action > 0:
            self.last_action_time = t

    # ----------------------------------------------------------

    def get_lambda_gamma(self):
        """
        Extract interpretable time dynamics:

        theta contains last 2 entries:
        - b1 → linear time effect
        - b2 → quadratic time effect

        We map them back to:
        λ (strength)
        γ (curvature)
        """

        b1 = self.theta_hat[-2]
        b2 = self.theta_hat[-1]

        learned_lambda = -b1

        if learned_lambda != 0:
            learned_gamma = b2 / learned_lambda
        else:
            learned_gamma = 0.0

        return learned_lambda, learned_gamma


# ============================================================
# DATA GENERATION (STATIC BANDIT ENVIRONMENT)
# ============================================================

def generate_bandit_data(
    T=1000,
    n_arms=2,
    base_d=5,
    context_dim=5,
    noise_std=0.1,
    seed=1234
):
    """
    Simulates a contextual bandit environment.

    Each timestep:
    - generate context per arm
    - compute reward = baseline + treatment + noise
    """

    np.random.seed(seed)

    d = n_arms * base_d

    # shared nonlinear structure across time
    X = np.random.normal(0, 1, size=(T, context_dim))
    w_baseline = np.random.normal(0, 1, size=context_dim)

    # random feature pool (used to build arm contexts)
    big_vector = np.random.normal(0, 1, size=n_arms * d * T)

    # true reward parameters (unknown to agent)
    theta_true = np.random.normal(0, 1, size=d)

    S = [[None for _ in range(n_arms)] for _ in range(T)]
    r = np.zeros((T, n_arms))
    baseline_values = np.zeros(T)

    for t in range(T):

        # shared baseline signal (same across arms)
        x_t = X[t]
        baseline = np.sin(x_t @ w_baseline)
        baseline_values[t] = baseline

        # sample raw features
        idx = np.random.choice(len(big_vector), size=n_arms * d, replace=False)
        sample = big_vector[idx]

        arm_features = []

        for a in range(n_arms):

            start = a * base_d
            end = (a + 1) * base_d

            raw = sample[start:end]

            # embed into full feature vector
            s_a = np.zeros(d)
            s_a[start:end] = raw

            arm_features.append(s_a)

        # store contexts
        for a in range(n_arms):
            S[t][a] = arm_features[a]

        # compute rewards
        for a in range(n_arms):

            start = a * base_d
            end = (a + 1) * base_d

            noise = noise_std * np.random.randn()
            treatment = sample[start:end] @ theta_true[start:end]

            r[t, a] = baseline + treatment + noise

    return S, r, theta_true, baseline_values, X, w_baseline


# ============================================================
# TRAINING SIMULATION (ENVIRONMENT + BANDIT TOGETHER)
# ============================================================

T = 2000
n_arms = 2
base_d = 5
context_dim = 5
noise_std = 0.01

d = n_arms * base_d
np.random.seed(1234)

# true hidden parameters
theta_true = np.random.normal(0, 1, size=d)
w_baseline = np.random.normal(0, 1, size=context_dim)

bandit = ActionCenteredThompson(d=d, N=n_arms)

regret = np.zeros(T)
theta_error = np.zeros(T)

last_action_time = -1  # environment state

# ============================================================
# MAIN LOOP
# ============================================================

for t in range(T):

    # random nonlinear baseline
    x_t = np.random.normal(0, 1, size=context_dim)
    baseline = np.sin(x_t @ w_baseline)

    # generate arm contexts
    context_features = []
    raw_features = []

    for a in range(n_arms):

        raw = np.random.normal(0, 1, size=base_d)

        s_a = np.zeros(d)
        start = a * base_d
        end = (a + 1) * base_d
        s_a[start:end] = raw

        context_features.append(s_a)
        raw_features.append(raw)

    # bandit chooses action
    action, a_bar, pi_t, s_bar, delta_t = bandit.choose_action(context_features, t)

    # compute true reward
    if action > 0:

        a_index = action - 1
        start = a_index * base_d
        end = (a_index + 1) * base_d

        noise = noise_std * np.random.randn()
        treatment = raw_features[a_index] @ theta_true[start:end]

        reward = baseline + treatment + noise

    else:
        reward = baseline

    # compute best possible reward (regret baseline)
    best_reward = baseline

    for a in range(n_arms):

        start = a * base_d
        end = (a + 1) * base_d

        treatment = raw_features[a] @ theta_true[start:end]
        candidate = baseline + treatment

        best_reward = max(best_reward, candidate)

    regret[t] = best_reward - reward

    # parameter estimation error
    theta_aug = np.concatenate([theta_true, [-1, 0]])  # placeholder time params
    theta_error[t] = np.linalg.norm(theta_aug - bandit.theta_hat)

    # update model
    bandit.update(s_bar, action, pi_t, reward, t)

    if action > 0:
        last_action_time = t

    print(f"t={t}, action={action}, pi={pi_t:.3f}, reward={reward:.3f}")


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n----- PARAMETER RECOVERY -----")
print("True theta:", theta_true)
print("Learned theta:", bandit.theta_hat)

l2 = np.linalg.norm(theta_true - bandit.theta_hat[:len(theta_true)])
print("L2 error:", l2)

print("Cumulative regret:", np.sum(regret))

learned_lambda, learned_gamma = bandit.get_lambda_gamma()
print("lambda:", learned_lambda)
print("gamma:", learned_gamma)


# ============================================================
# PLOTS
# ============================================================

# ============================================================
# PLOTS (CLEAN + LABELED)
# ============================================================

# convert regret to average regret
avg_regret = np.cumsum(regret) / np.arange(1, T + 1)

plt.figure(figsize=(12, 5))

# -----------------------------
# Theta estimation error
# -----------------------------
plt.subplot(1, 2, 1)
plt.plot(theta_error, linewidth=2)
plt.title("Theta Estimation Error Over Time")
plt.xlabel("Time Step")
plt.ylabel("L2 Error")
plt.grid(True, alpha=0.3)

# -----------------------------
# Average regret
# -----------------------------
plt.subplot(1, 2, 2)
plt.plot(avg_regret, linewidth=2)
plt.title("Average Regret Over Time")
plt.xlabel("Time Step")
plt.ylabel("Average Regret")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()