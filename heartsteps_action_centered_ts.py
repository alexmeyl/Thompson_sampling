import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# THOMPSON SAMPLING BANDIT (ACTION-DEPENDENT VERSION)
# ============================================================

class ActionCenteredThompson:
    """
    This is a contextual bandit using Thompson Sampling.

    Core idea:
    - We learn a linear model: reward ≈ theta^T * features
    - Instead of greedy selection, we sample theta from uncertainty
    - Then choose actions based on sampled model (exploration + exploitation)
    """

    def __init__(self, d, pi_min=0.2, pi_max=0.8, v=0.5, ridge=10.0):

        # feature dimension (extra 3 added for user burden features)
        self.d = d + 3

        # clamp probability of taking action
        self.pi_min = pi_min
        self.pi_max = pi_max

        # controls randomness of parameter sampling
        self.v = v

        # Bayesian linear regression prior
        self.B = ridge * np.eye(self.d)   # precision matrix (confidence)
        self.b = np.zeros(self.d)          # reward-weighted feature sum
        self.theta_hat = np.zeros(self.d)  # current parameter estimate

        # user history: stores past actions per user
        self.history = {}

    def sample_theta(self):
        """
        Sample model parameters from posterior uncertainty.

        Intuition:
        - If a feature is uncertain → higher variance
        - We randomly sample a plausible model
        """
        diag = np.clip(np.diag(self.B), 1e-6, None)
        std = self.v / np.sqrt(diag)
        return self.theta_hat + std * np.random.randn(self.d)

    def compute_burden(self, user_id, current_day):
        """
        Measures recent exposure of a user to actions.

        Idea:
        - If user received many actions recently → "burden" increases
        - We count actions in last 3 days
        """
        history = self.history.get(user_id, [])
        return sum(
            1 for day, action in history
            if action > 0 and 0 < (current_day - day) <= 3
        )

    def choose_action(self, s, user_id, current_day):
        """
        Decide whether to take action (1) or not (0).

        Steps:
        1. Build augmented feature vector (includes user burden + time)
        2. Sample a model (Thompson sampling)
        3. Predict reward distribution
        4. Convert to probability of positive reward
        5. Sample final action
        """

        # how often user recently received actions
        N_dk = self.compute_burden(user_id, current_day)

        # normalize time and burden
        d_scaled = current_day / 50.0
        N_scaled = N_dk / 5.0

        # extended feature vector (original + interaction effects)
        s_bar = np.concatenate([s, [N_scaled, d_scaled, N_scaled * d_scaled]])

        if not np.all(np.isfinite(s_bar)):
            return 0, 0.5, s_bar, N_dk

        # sample plausible model from uncertainty
        theta = self.sample_theta()

        # predicted mean reward
        mean = s_bar @ theta

        # estimate uncertainty (variance of prediction)
        try:
            B_inv = np.linalg.inv(self.B)
            var = s_bar @ B_inv @ s_bar
        except np.linalg.LinAlgError:
            var = 1.0

        if not np.isfinite(var):
            var = 1.0

        # convert variance into standard deviation
        std = np.sqrt(max(self.v**2 * var, 1e-8))

        # probability reward > 0 under Gaussian assumption
        p_pos = 1 - norm.cdf(0, loc=mean, scale=std)

        # clip to avoid extreme probabilities
        pi_t = np.clip(p_pos, self.pi_min, self.pi_max)

        # sample action from Bernoulli(pi_t)
        action = int(np.random.rand() < pi_t)
        print(f"User {user_id}, Day {current_day}: p_pos={p_pos:.3f}, pi_t={pi_t:.3f}, action={action}, N_dk={N_dk}")
        return action, pi_t, s_bar, N_dk

    def update(self, s_bar, action, pi_t, reward, logged_action, mu_at):
        """
        Update model using observed data.

        Only update if:
        - we actually took the same action as logged system
        - reward is valid

        This avoids bias from mismatched logging policy.
        """

        if action != logged_action or not np.isfinite(reward):
            return None

        indicator = 1 if action > 0 else 0

        # Bayesian linear regression update
        self.B += pi_t * (1 - pi_t) * np.outer(s_bar, s_bar)
        self.b += s_bar * (indicator - pi_t) * reward

        self.theta_hat = np.linalg.solve(self.B, self.b)

        # importance-sampled reward estimate
        return reward * (pi_t / mu_at)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def compute_study_day(df):
    """
    Converts timestamps into "day since first interaction per user".

    Why:
    - removes absolute time effects
    - normalizes users starting at different times
    """
    dt = pd.to_datetime(df["sugg.decision.utime"], errors="coerce")
    day = dt.dt.normalize()

    first = day.groupby(df["user.index"]).transform("min")
    study_day = (day - first).dt.days

    # fallback if timestamps missing
    fallback = df.groupby("user.index").cumcount()
    return study_day.fillna(fallback).astype(int)


class FeatureEncoder:
    """
    Builds feature statistics and vocabulary for encoding rows.
    """

    def __init__(self, top_k_weather=5):
        self.top_k_weather = top_k_weather
        self.weather_vocab = []

        self.numeric_cols = [
            "dec.temperature",
            "dec.windspeed",
            "dec.precipitation.chance",
            "dec.snow",
        ]

        self.means = {}
        self.stds = {}

    def fit(self, df):
        """
        Learn:
        - most common weather categories
        - mean/std for normalization of numeric features
        """

        weather = df["dec.weather.condition"].astype(str).str.lower().str.strip()
        counts = weather.value_counts()

        self.weather_vocab = [w for w in counts.index if w != "nan"][:self.top_k_weather]

        for c in self.numeric_cols:
            vals = pd.to_numeric(df[c], errors="coerce")

            # mean-centering baseline
            self.means[c] = vals.mean() if np.isfinite(vals.mean()) else 0.0

            # scaling factor
            self.stds[c] = vals.std() if vals.std() > 0 else 1.0

        return self


def get_features(row, enc):
    """
    Converts a single row into a numeric feature vector.

    Includes:
    - user activity flags
    - location encoding
    - weather encoding
    - normalized numeric environmental data
    """

    def safe(x, default=0.0):
        try:
            return float(x) if not pd.isna(x) else default
        except:
            return default

    # user behavior signals
    feats = [
        safe(row.get("tag.active")),
        safe(row.get("tag.indoor")),
        safe(row.get("tag.outdoor")),
        safe(row.get("tag.outdoor_snow")),
    ]

    # location encoding (one-hot style)
    loc = str(row.get("dec.location.category", "other")).lower()
    feats += [
        loc == "home",
        loc == "work",
        loc not in ["home", "work"],
    ]

    # weather one-hot encoding
    weather = str(row.get("dec.weather.condition", "other")).lower()

    for w in enc.weather_vocab:
        feats.append(weather == w)

    feats.append(weather not in enc.weather_vocab)

    # numeric normalization (z-score style)
    for c in enc.numeric_cols:
        val = pd.to_numeric(row.get(c), errors="coerce")

        if not np.isfinite(val):
            val = enc.means[c]

        if not np.isfinite(val):
            val = 0.0

        feats.append((val - enc.means[c]) / enc.stds[c])

    return np.array(feats)


def get_reward(row):
    """
    Reward = change in steps after intervention.

    Positive reward = user improved
    Negative reward = user declined
    """

    post = pd.to_numeric(row.get("jbsteps30.zero"), errors="coerce")
    pre = pd.to_numeric(row.get("jbsteps30pre.zero"), errors="coerce")

    if pd.isna(post) or pd.isna(pre):
        return None

    return post - pre


# ============================================================
# TRAINING LOOP
# ============================================================

def train_bandit(csv_path):
    """
    Main training loop:

    For each user interaction:
    1. Extract features
    2. Bandit chooses action
    3. Compare with logged system action
    4. Compute reward
    5. Update model if valid
    """

    df = pd.read_csv(csv_path, low_memory=False)

    df["_dt"] = pd.to_datetime(df["sugg.decision.utime"], errors="coerce")
    df = df.sort_values(["user.index", "_dt"]).reset_index(drop=True)

    df["day"] = compute_study_day(df)

    enc = FeatureEncoder().fit(df)
    d = len(get_features(df.iloc[0], enc))
    bandit = ActionCenteredThompson(d)

    ips_total = 0
    matched = 0

    for _, row in df.iterrows():

        s = get_features(row, enc)
        user = row["user.index"]
        day = int(row["day"])

        # model decision
        action, pi_t, s_bar, _ = bandit.choose_action(s, user, day)

        logged = row.get("send")
        if pd.isna(logged):
            continue

        logged = int(bool(logged))

        # store behavior history
        bandit.history.setdefault(user, []).append((day, logged))

        reward = get_reward(row)
        if reward is None:
            continue

        mu = 0.6 if logged else 0.4

        # only update if actions match (important for unbiased learning)
        if action == logged:
            ips = bandit.update(s_bar, action, pi_t, reward, logged, mu)

            if ips is not None and np.isfinite(ips):
                ips_total += ips
                matched += 1

    policy_value = ips_total / max(matched, 1)

    print("Matched:", matched)
    print("IPS total:", ips_total)
    print("Policy value:", policy_value)


if __name__ == "__main__":
    train_bandit("suggestions.csv")