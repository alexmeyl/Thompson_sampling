# # Load the three CSV files
# import pandas as pd

# # Load gfsteps.csv
# gfsteps = pd.read_csv('gfsteps.csv')
# print("gfsteps.csv loaded - Shape:", gfsteps.shape)
# print(gfsteps.head())
# print("\n")

# # Load jbsteps.csv
# jbsteps = pd.read_csv('jbsteps.csv')
# print("jbsteps.csv loaded - Shape:", jbsteps.shape)
# print(jbsteps.head())
# print("\n")

# # Load suggestions.csv
# suggestions = pd.read_csv('suggestions.csv')
# print("suggestions.csv loaded - Shape:", suggestions.shape)
# print(suggestions.head())
# print("\n") 



import numpy as np
from scipy.stats import norm

class ActionCenteredThompson:
    def __init__(self, d, N, pi_min=0.2, pi_max=0.8, v=1.0):
        
        """
        d      : dimension of feature vector s_{t,a}
        N      : number of non-zero actions
        pi_min : minimum probability of taking non-zero action
        pi_max : maximum probability of taking non-zero action
        v      : Thompson sampling exploration parameter
        """

        self.d = d
        self.N = N
        self.pi_min = pi_min
        self.pi_max = pi_max
        self.v = v

        # Algorithm state
        self.B = np.eye(d)        # Design identity matrix
        self.b = np.zeros(d)      # Response vector
        self.theta_hat = np.zeros(d)

    # ----------------------------------------------------------

    def sample_theta(self):
        """Sample θ' ~ N(theta_hat, v^2 * B^{-1})"""
        cov = self.v ** 2 * np.linalg.inv(self.B)
        return np.random.multivariate_normal(self.theta_hat, cov)

    # ----------------------------------------------------------

    def choose_action(self, context_features):
        """
        context_features: list of feature vectors s_{t,a} for a=1..N
        returns: chosen action at, pseudo-action a_bar, probability pi_t
        """

        # ----- Step 4: Sample theta prime -----
        theta_prime = self.sample_theta()

        # ----- Step 5: Choose best non-zero action -----
        scores = [s @ theta_prime for s in context_features] # for each action dot product of s_t_a and theta_prime
        a_bar = int(np.argmax(scores))   # index 0...(N-1)

        s_bar = context_features[a_bar]

        # ----- Step 6: Compute probability π_t -----

        mean = s_bar @ self.theta_hat 
        var = s_bar @ np.linalg.inv(self.B) @ s_bar
        std = np.sqrt(self.v ** 2 * var)

        # P(s^T θ̃ > 0)
        p_positive = 1 - norm.cdf(0, loc=mean, scale=std)

        pi_t = np.clip(p_positive, self.pi_min, self.pi_max)

        # ----- Step 7: Randomize between 0 and a_bar -----

        if np.random.rand() < pi_t:
            action = a_bar + 1      # convert to 1..N
        else:
            action = 0

        return action, a_bar, pi_t

    # ----------------------------------------------------------

    def update(self, s_bar, action, pi_t, reward):
        """
        Perform update after observing reward r_t(a_t)
        s_bar  : feature vector of pseudo action a_bar
        action : actual action taken (0 or 1..N)
        pi_t   : probability of non-zero action
        reward : observed reward r_t(a_t)
        """

        indicator = 1 if action > 0 else 0

        # ----- Step 8: Update B and b -----

        self.B += pi_t * (1 - pi_t) * np.outer(s_bar, s_bar)

        self.b += s_bar * (indicator - pi_t) * reward

        # Update theta_hat
        self.theta_hat = np.linalg.inv(self.B) @ self.b


def build_features(context, N):
    # context: vector s_bar_t from paper
    features = []

    for a in range(N):
        # Example: simple shared features
        features.append(context)

    return features



# 2 arms, generte context features for each arm for T time steps each
# 5 elements in the context for each arm
# T=100 number of time steps
# S - list of context features vectors for each arm at each time step

# Multivariate normal vector with a 1000 elements, mean = 0, vaer =1
# Take 10 elements, first 5 - context for arm 1, next 5 - context for arm 2 

# Generate true theta vector with 5 elements, mean=0, var=1
# tratement effects = S_t_a @ theta_true + 0.1*np.rand(), we will have: 100*2 treatment effects for each arm for each time step

import numpy as np

# set random seed for reproducibility
np.random.seed(1234)

# ----- Parameters -----
T = 100          # number of time steps
n_arms = 2
d = 5            # features per arm

# ----- Generate big multivariate normal pool -----
big_vector = np.random.normal(loc=0, scale=1, size=n_arms*d*T)

# ----- Generate true theta -----
theta_true = np.random.normal(0, 1, size=d)

# ----- Containers -----
# S_t_a will be the context vector for time t and arm a
S = [[None for _ in range(n_arms)] for _ in range(T)]

# True treatment effects r_t_a
r = np.zeros((T, n_arms))

# ----- Build contexts and rewards -----
for t in range(T):

    # sample 10 elements from the big vector
    idx = np.random.choice(len(big_vector), size=10, replace=False)

    sample = big_vector[idx]

    # first 5 → arm 1 context
    s1 = sample[:5]

    # next 5 → arm 2 context
    s2 = sample[5:]

    S[t][0] = s1
    S[t][1] = s2

    # ----- treatment effects -----
    noise1 = 0.1 * np.random.randn()
    noise2 = 0.1 * np.random.randn()

    r[t,0] = s1 @ theta_true + noise1
    r[t,1] = s2 @ theta_true + noise2


# ----- Quick check -----
print("Theta true:", theta_true)
print("Example context arm1 t=0:", S[0][0])
print("Example reward arm1 t=0:", r[0,0])

from thompson_sampling_centered_reward import ActionCenteredThompson

# ----- Create bandit instance -----
bandit = ActionCenteredThompson(d=d, N=n_arms)

# ----- Run algorithm -----
for t in range(T):

    context_features = S[t]   # list of 2 vectors

    # call method on the OBJECT
    action, a_bar, pi_t = bandit.choose_action(context_features)

    # get reward for chosen action
    reward = r[t, action-1] if action > 0 else 0

    # update 
    bandit.update(context_features[a_bar], action, pi_t, reward)

    print(f"t={t}, action={action}, pi={pi_t:.3f}, reward={reward:.3f}")


# ----- Final check -----
print("\n----- Parameter Recovery -----")
print("True theta:   ", theta_true)
print("Learned theta:", bandit.theta_hat)
l2 = np.linalg.norm(theta_true - bandit.theta_hat)
rel = l2 / np.linalg.norm(theta_true)
print("L2 error:       {:.4f}".format(l2))
print("Relative error: {:.2%}".format(rel))