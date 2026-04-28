import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# ============================================================
# THOMPSON SAMPLING BANDIT (ACTION SELECTION)
# ============================================================

class ActionCenteredThompson:
    """
    Contextual bandit using Thompson Sampling.

    Goal:
    Learn a linear model:
        reward ≈ theta^T * context

    Then:
    - sample plausible theta
    - pick best action under sampled model
    - convert score into probability
    - randomly decide (exploration)
    """

    def __init__(self, d, N, pi_min=0.2, pi_max=0.8, v=1.0):

        # feature dimension
        self.d = d
        self.N = N

        # action probability clipping
        self.pi_min = pi_min
        self.pi_max = pi_max

        # exploration strength (higher = more randomness)
        self.v = v

        # Bayesian linear regression state
        self.B = np.eye(d)        # confidence matrix
        self.b = np.zeros(d)      # reward accumulation
        self.theta_hat = np.zeros(d)

    # ----------------------------------------------------------

    def sample_theta(self):
        """
        Sample a plausible parameter vector from uncertainty.

        If model is uncertain → covariance is large → more exploration
        """
        cov = self.v ** 2 * np.linalg.inv(self.B)
        return np.random.multivariate_normal(self.theta_hat, cov)

    # ----------------------------------------------------------

    def choose_action(self, context_features):
        """
        Decide action for current timestep.

        Steps:
        1. sample model (theta')
        2. score each arm
        3. pick best arm under sampled model
        4. compute probability of taking action
        5. sample final action (0 or a_bar)
        """

        # sample model
        theta_prime = self.sample_theta()

        # score each action
        scores = [s @ theta_prime for s in context_features]

        # best action under sampled model
        a_bar = int(np.argmax(scores))
        s_bar = context_features[a_bar]

        # predicted mean reward
        mean = s_bar @ self.theta_hat

        # uncertainty of prediction
        var = s_bar @ np.linalg.inv(self.B) @ s_bar
        std = np.sqrt(self.v ** 2 * var)

        # probability reward > 0
        p_positive = 1 - norm.cdf(0, loc=mean, scale=std)

        # clip for stability
        pi_t = np.clip(p_positive, self.pi_min, self.pi_max)

        # stochastic decision
        action = a_bar + 1 if np.random.rand() < pi_t else 0

        return action, a_bar, pi_t

    # ----------------------------------------------------------

    def update(self, s_bar, action, pi_t, reward):
        """
        Update model after observing reward.

        Only updates belief about theta using:
        - Bayesian linear regression style update
        - importance weighting via pi_t
        """

        indicator = 1 if action > 0 else 0

        # confidence update
        self.B += pi_t * (1 - pi_t) * np.outer(s_bar, s_bar)

        # reward update
        self.b += s_bar * (indicator - pi_t) * reward

        # update parameter estimate
        self.theta_hat = np.linalg.inv(self.B) @ self.b


# ============================================================
# DATA GENERATION
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
    Simulates contextual bandit environment.

    Each timestep:
    - generate arm contexts
    - compute reward = baseline + treatment + noise
    """

    np.random.seed(seed)

    d = n_arms * base_d

    # shared nonlinear environment signal
    X = np.random.normal(0, 1, size=(T, context_dim))
    w_baseline = np.random.normal(0, 1, size=context_dim)

    # random feature pool
    big_vector = np.random.normal(0, 1, size=n_arms * d * T)

    # true hidden parameter (unknown to agent)
    theta_true = np.random.normal(0, 1, size=d)

    S = [[None for _ in range(n_arms)] for _ in range(T)]
    r = np.zeros((T, n_arms))
    baseline_values = np.zeros(T)

    for t in range(T):

        # nonlinear shared baseline signal
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
# TRAINING LOOP
# ============================================================

T = 2000
n_arms = 2
base_d = 5
context_dim = 5
noise_std = 0.1

d = n_arms * base_d
np.random.seed(1234)

# generate environment
S, r, theta_true, baseline_values, X, w_baseline = generate_bandit_data(
    T=T, n_arms=n_arms, base_d=base_d, context_dim=context_dim, noise_std=noise_std
)

bandit = ActionCenteredThompson(d=d, N=n_arms)

regret = np.zeros(T)
theta_error = np.zeros(T)

# ============================================================
# MAIN LOOP
# ============================================================

for t in range(T):

    context_features = S[t]

    # bandit decision
    action, a_bar, pi_t = bandit.choose_action(context_features)

    # reward (action 0 = baseline)
    reward = r[t, action - 1] if action > 0 else baseline_values[t]

    # best possible reward
    best_reward = max(baseline_values[t], np.max(r[t]))
    regret[t] = best_reward - reward

    # parameter error
    theta_error[t] = np.linalg.norm(theta_true - bandit.theta_hat)

    # update model
    bandit.update(context_features[a_bar], action, pi_t, reward)

    print(f"t={t}, action={action}, pi={pi_t:.3f}, reward={reward:.3f}")


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n----- Parameter Recovery -----")
print("True theta:", theta_true)
print("Learned theta:", bandit.theta_hat)

l2 = np.linalg.norm(theta_true - bandit.theta_hat)
rel = l2 / np.linalg.norm(theta_true)

print("L2 error:", l2)
print("Relative error:", rel)
print("Cumulative regret:", np.sum(regret))
print("Final theta error:", theta_error[-1])


# ============================================================
# PLOTS (MATCHING CLEAN STYLE)
# ============================================================

avg_regret = np.cumsum(regret) / np.arange(1, T + 1)

plt.figure(figsize=(12, 5))

# theta error
plt.subplot(1, 2, 1)
plt.plot(theta_error, linewidth=2)
plt.title("Theta Estimation Error Over Time")
plt.xlabel("Time Step")
plt.ylabel("L2 Error")
plt.grid(True, alpha=0.3)

# regret
plt.subplot(1, 2, 2)
plt.plot(avg_regret, linewidth=2)
plt.title("Average Regret Over Time")
plt.xlabel("Time Step")
plt.ylabel("Average Regret")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()