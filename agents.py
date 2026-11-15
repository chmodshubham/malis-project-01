"""Agents for Fishing Season (MALIS, Project 1).

This file is a skeleton. Write all your agents in it.

Do NOT modify the classes Agent and ContextualAgent, nor the function play():
they define how every agent interacts with the simulator, and they are used
to run your agents during grading.

How an agent works
------------------
For each season, play(agent, bay) calls:

    agent.reset()                     once, before the first day
    spot = agent.choose()             every day: return a spot, 0 to 4
    agent.update(spot, catch)         every day: the catch you observed

An agent only learns through update(). It must not look at the bay itself.

Contextual agents (Part IV, optional) follow the same pattern, but receive the
morning conditions x (numpy array [wind_kmh, waves_m, tide_m]):

    spot = agent.choose(x)
    agent.update(x, spot, catch)

Example
-------
    from fishing import FishingBay
    from agents import RandomAgent, play

    bay = FishingBay(season=1, seed=0)
    total = play(RandomAgent(seed=0), bay)
"""
import numpy as np

from fishing import ContextualBay, FishingBay

# ---------------------------------------------------------------------------
# Provided: do not modify
# ---------------------------------------------------------------------------


class Agent:
    """Base class for the agents of Parts II and III.

    Parameters
    ----------
    n_spots : int
        Number of fishing spots (5 in the game).
    seed : int or None
        Seed of the agent's own random generator, self.rng. Use self.rng for
        any random decision, so that results can be reproduced.
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self):
        """Return the spot to fish today, an int between 0 and n_spots - 1."""
        raise NotImplementedError

    def update(self, spot, catch):
        """Called after each day with the spot chosen and the catch observed (kg)."""


class ContextualAgent:
    """Base class for the agents of Part IV (contextual bay).

    Same as Agent, except that choose() and update() also receive x, the
    conditions of the day: numpy array [wind_kmh, waves_m, tide_m].
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self, x):
        """Return the spot to fish today given the conditions x."""
        raise NotImplementedError

    def update(self, x, spot, catch):
        """Called after each day with the conditions, the spot chosen and the catch observed."""


def play(agent, bay):
    """Play one full season of bay with agent. Returns the total catch (kg).

    Works with FishingBay and Agent, or with ContextualBay and ContextualAgent.
    The bay is reset first, so the same bay can be played by several agents.
    """
    contextual = isinstance(bay, ContextualBay)
    if contextual and not isinstance(agent, ContextualAgent):
        raise TypeError("a ContextualBay must be played by a ContextualAgent")
    if not contextual and not isinstance(agent, Agent):
        raise TypeError("a FishingBay must be played by an Agent")
    bay.reset()
    agent.reset()
    while not bay.done:
        if contextual:
            x = bay.observe_array()
            spot = int(agent.choose(x))
            catch = bay.fish(spot)
            agent.update(x, spot, catch)
        else:
            spot = int(agent.choose())
            catch = bay.fish(spot)
            agent.update(spot, catch)
    return bay.total


# ---------------------------------------------------------------------------
# Provided example agents (baselines)
# ---------------------------------------------------------------------------


class RandomAgent(Agent):
    """Chooses a spot uniformly at random every day. Learns nothing."""

    def choose(self):
        return int(self.rng.integers(self.n_spots))


class RandomContextualAgent(ContextualAgent):
    """Ignores the conditions and chooses a spot uniformly at random every day."""

    def choose(self, x):
        return int(self.rng.integers(self.n_spots))


# ---------------------------------------------------------------------------
# Your agents
# ---------------------------------------------------------------------------
#
# Everything below this line was written by our team (the provided code above
# is unchanged). Notation follows Sutton & Barto, Chapter 2:
#   Q(a)  estimated value (average catch) of spot a
#   N(a)  number of times spot a has been chosen
#   t     number of days already fished in the current season
# Every random decision uses self.rng (seeded in the constructor), so a run
# can be reproduced exactly by giving the same seed.


def _argmax_random_tie(values, rng):
    """Index of the maximum of `values`, breaking ties uniformly at random.

    Why: np.argmax always returns the FIRST maximum. With equal estimates (for
    example at the start of a season, when every Q is the same) that would make
    every agent always start at spot 0, which is an arbitrary bias.
    """
    values = np.asarray(values, dtype=float)
    best = np.flatnonzero(values == values.max())
    return int(rng.choice(best))


# ===========================================================================
# Task 4 (part 1): human strategies from Part II, written as precise rules
# ===========================================================================
#
# We abstracted two of our team's strategies into rules that can be applied to
# any bay (see Description.txt). The third one (kanakara: always move to a
# different spot) never exploits, so it behaves like RandomAgent and was not
# worth coding again.
#
# Which one is "best"? We simulated both rules on 1000 random bays per season
# and measured the fraction of the best possible catch they get (our Task 3
# measure, see the README). kumarsh's rule wins in Seasons 1 and 3 and ties in
# Season 2 (experiment.ipynb, section 1.1), so it is HumanRuleAgent;
# bouazza's rule is kept as ThresholdRuleAgent for the comparison.


class ThresholdRuleAgent(Agent):
    """Second human strategy (bouazza's threshold rule), kept to compare with HumanRuleAgent.

    The rule, precise enough to apply by hand. Every evening, after seeing
    today's catch c at spot s, decide tomorrow's spot with the first line that
    applies:

    0. Day 1: fish at `start_spot` (2 = Seagrass Bed, "the middle of the bay").
    1. Rapid search in progress: go to the next spot of the search list.
    2. Hot spot (c > `high`, 7 kg): stay at s.
    3. Collapse (c < `collapse`, 2 kg, at a spot whose recent average was
       above `high`): the fish have probably moved (Season 3). Start a rapid
       search: visit every other spot once, least recently visited first.
    4. Forgotten spot (some spot not visited for `revisit_gap` = 8 days, or
       never visited): test it, the one not visited for the longest time.
    5. Bad day (c < `low`, 6 kg): move to the other spot with the best recent
       average (average of its last `memory` = 3 catches).
    6. Medium day (low <= c <= high): if today was a test of another spot, go
       back to the spot with the best recent average; otherwise stay.

    Why a recent average (last 3 catches) and not the average of all catches:
    it is what a human remembers, and it lets the rule forget the fish
    populations of before the change in Season 3.

    Parameters
    ----------
    low, high : float
        Leave below `low` kg, stay above `high` kg (Rules 1 and 2 of the notes).
    revisit_gap : int
        Days after which an unvisited spot is tested again (Rule 3 of the notes).
    collapse : float
        A catch below this at a reliable spot triggers a rapid search (Rule 4).
    memory : int
        Number of recent catches averaged to judge a spot.
    start_spot : int
        Spot fished on day 1.
    """

    def __init__(self, n_spots=5, seed=None, low=6.0, high=7.0, revisit_gap=8,
                 collapse=2.0, memory=3, start_spot=2):
        super().__init__(n_spots, seed)
        self.low, self.high = low, high
        self.revisit_gap = revisit_gap
        self.collapse = collapse
        self.memory = memory
        self.start_spot = start_spot

    def reset(self):
        self.t = 0                                   # days fished so far
        self.next_spot = self.start_spot             # decided the evening before
        self.last_seen = np.full(self.n_spots, -10**9)  # day of last visit (never = far past)
        self.recent = [[] for _ in range(self.n_spots)]  # last `memory` catches per spot
        self.search = []                             # spots left to visit in a rapid search
        self.home = self.start_spot                  # spot we were exploiting before a test

    def _recent_mean(self, spot):
        """Average of the last catches at `spot` (+inf if never visited, so it gets tested)."""
        r = self.recent[spot]
        return np.mean(r) if r else np.inf

    def choose(self):
        return int(self.next_spot)

    def update(self, spot, catch):
        # Was today a test of a spot other than our current "home" spot?
        was_test = spot != self.home
        # Remember the reliability of this spot BEFORE adding today's catch
        # (needed by Rule 3: was it a good spot until today?).
        was_reliable = bool(self.recent[spot]) and np.mean(self.recent[spot]) > self.high

        # --- memory update ---
        self.recent[spot] = (self.recent[spot] + [catch])[-self.memory:]
        self.last_seen[spot] = self.t
        self.t += 1

        others = [k for k in range(self.n_spots) if k != spot]
        means = np.array([self._recent_mean(k) for k in range(self.n_spots)])
        # Spots not visited for revisit_gap days (never-visited spots included)
        forgotten = [k for k in others if self.t - self.last_seen[k] >= self.revisit_gap]

        # --- the rule (first line that applies) ---
        if self.search:                                         # 1. rapid search
            nxt = self.search.pop(0)
        elif catch > self.high:                                 # 2. hot spot: stay
            nxt = spot
        elif catch < self.collapse and was_reliable:            # 3. collapse
            self.search = sorted(others, key=lambda k: self.last_seen[k])
            nxt = self.search.pop(0)
        elif forgotten:                                         # 4. test a forgotten spot
            nxt = min(forgotten, key=lambda k: self.last_seen[k])
        elif catch < self.low:                                  # 5. bad day: best other spot
            nxt = max(others, key=lambda k: means[k])
        elif was_test:                                          # 6a. medium test: go back to best
            nxt = _argmax_random_tie(means, self.rng)
        else:                                                   # 6b. medium day at home: stay
            nxt = spot

        # The "home" spot is the one we fish when we are not testing:
        # a spot becomes home when we decide to stay at it.
        if nxt == spot:
            self.home = spot
        elif not self.search and nxt not in forgotten:
            self.home = nxt
        self.next_spot = nxt


class HumanRuleAgent(Agent):
    """Task 4: the best human strategy of our team (kumarsh's "sample, then stick").

    The rule:
    1. Days 1 to n_spots: fish each spot once (in a random order).
    2. Afterwards: fish the spot with the best average catch so far
       (average of ALL catches at that spot).
    3. Except: if yesterday's catch was below `low` (5 kg), do not fish that
       spot today; take the best of the other spots.

    The "centre bias" of the original notes (prefer Seagrass Bed and Deep
    Channel) was dropped: spot names carry no information in a random bay.

    It is in fact "explore first, then greedy" (Section 2.2 with epsilon = 0
    after one sample of each arm) plus a leave-on-bad-day rule.
    """

    def __init__(self, n_spots=5, seed=None, low=5.0):
        super().__init__(n_spots, seed)
        self.low = low

    def reset(self):
        self.order = list(self.rng.permutation(self.n_spots))  # sampling phase order
        self.sum = np.zeros(self.n_spots)
        self.N = np.zeros(self.n_spots)
        self.last = None                                         # (spot, catch) of yesterday

    def choose(self):
        if self.order:                                  # 1. sample every spot once
            return int(self.order.pop(0))
        means = self.sum / np.maximum(self.N, 1)        # 2. best average so far
        if self.last is not None and self.last[1] < self.low:
            means[self.last[0]] = -np.inf               # 3. not the spot that just failed
        return _argmax_random_tie(means, self.rng)

    def update(self, spot, catch):
        self.sum[spot] += catch
        self.N[spot] += 1
        self.last = (spot, catch)


# ===========================================================================
# Task 4 (part 2): methods from Sutton & Barto, Sections 2.1 to 2.7
# ===========================================================================


class EpsilonGreedyAgent(Agent):
    """Epsilon-greedy action-value method (Sections 2.2 to 2.5).

    Memory: Q (estimated average catch of each spot) and N (visits per spot).

    Choice: with probability `epsilon`, a spot uniformly at random (explore);
    otherwise the spot with the largest Q (exploit, ties broken at random).

    Update, after catching R at spot a (incremental form, Section 2.3):
        N(a) <- N(a) + 1
        Q(a) <- Q(a) + step * (R - Q(a))
    with step = 1 / N(a) if alpha is None: Q is the sample average (eq. 2.1);
    or step = alpha (constant step size, Section 2.4): Q is an exponential
    recency-weighted average, which forgets old catches. Useful in Season 3,
    where the averages change on day 16.

    Parameters
    ----------
    epsilon : float in [0, 1]
        Probability of exploring.
    alpha : float in (0, 1] or None
        Constant step size, or None for sample averages.
    q0 : float
        Initial estimate Q1(a) of every spot. With sample averages it only
        matters until the first visit. A large q0 is "optimistic" (Section 2.5).
    """

    def __init__(self, n_spots=5, seed=None, epsilon=0.1, alpha=None, q0=0.0):
        super().__init__(n_spots, seed)
        self.epsilon, self.alpha, self.q0 = epsilon, alpha, q0

    def reset(self):
        self.Q = np.full(self.n_spots, float(self.q0))
        self.N = np.zeros(self.n_spots)

    def choose(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_spots))  # explore
        return _argmax_random_tie(self.Q, self.rng)       # exploit

    def update(self, spot, catch):
        self.N[spot] += 1
        step = 1.0 / self.N[spot] if self.alpha is None else self.alpha
        self.Q[spot] += step * (catch - self.Q[spot])


class OptimisticGreedyAgent(EpsilonGreedyAgent):
    """Optimistic initial values (Section 2.5): greedy (epsilon = 0) with a large q0.

    Every spot starts with Q = q0 (15 kg, above any realistic average: the true
    averages are between 2 and 10 kg). Each catch is "disappointing", so Q of
    the visited spot drops below the untried ones and the agent tries every
    spot early. Exploration is driven by the initial values, not by chance.

    With sample averages (alpha=None, our default) q0 is forgotten at the
    first visit: the agent tries each spot once, then is greedy. With a
    constant step size `alpha` the optimism fades away slowly (weight
    (1 - alpha)^N on q0, eq. 2.6), so spots are re-tried several times.
    """

    def __init__(self, n_spots=5, seed=None, q0=15.0, alpha=None):
        super().__init__(n_spots, seed, epsilon=0.0, alpha=alpha, q0=q0)


class UCBAgent(Agent):
    """Upper-Confidence-Bound action selection (Section 2.6, eq. 2.8).

    Memory: Q, N as in EpsilonGreedyAgent, and t, the number of days fished.

    Choice: any spot never tried first (N(a) = 0 counts as a maximizing
    action); then
        A = argmax_a  Q(a) + c * sqrt( ln(t) / N(a) )
    The square-root term measures the uncertainty of Q(a): it shrinks when a
    is tried and grows slowly when other spots are tried. So UCB explores the
    spots that might be the best, not uniformly as epsilon-greedy does.

    Parameters
    ----------
    c : float > 0
        Degree of exploration. The bonus is in kg, so c must be of the order
        of the catch noise (standard deviation 2.5 kg): c = 0 is pure greedy.
    alpha : float or None
        Constant step size for Q (Section 2.4) or None for sample averages.
        A constant alpha (e.g. 0.5) makes Q follow the change of Season 3:
        after the change, the catches of a spot that collapsed quickly pull
        its Q down, instead of being averaged with 15 good days.
    """

    def __init__(self, n_spots=5, seed=None, c=2.0, alpha=None):
        super().__init__(n_spots, seed)
        self.c, self.alpha = c, alpha

    def reset(self):
        self.Q = np.zeros(self.n_spots)
        self.N = np.zeros(self.n_spots)
        self.t = 0

    def choose(self):
        untried = np.flatnonzero(self.N == 0)
        if untried.size:                         # N(a) = 0: a is maximizing
            return int(self.rng.choice(untried))
        bonus = self.c * np.sqrt(np.log(self.t) / self.N)
        return _argmax_random_tie(self.Q + bonus, self.rng)

    def update(self, spot, catch):
        self.t += 1
        self.N[spot] += 1
        step = 1.0 / self.N[spot] if self.alpha is None else self.alpha
        self.Q[spot] += step * (catch - self.Q[spot])

# ===========================================================================
# Task 6 (optional, Part IV): contextual agents
# ===========================================================================
#
# The conditions x = [wind_kmh, waves_m, tide_m] are first scaled to [0, 1]
# (dividing by 40, 3 and 2, their maximal values) so that the three features
# have the same scale for KNN distances and for the ridge penalty. A constant
# 1 is added for the intercept of the linear regression.

_X_SCALE = np.array([40.0, 3.0, 2.0])


def _features(x):
    """[1, wind/40, waves/3, tide/2]: intercept + conditions scaled to [0, 1]."""
    return np.concatenate(([1.0], np.asarray(x, dtype=float) / _X_SCALE))


class ContextualWrapper(ContextualAgent):
    """Lets a Part III Agent play a ContextualBay by ignoring the conditions x.

    Used to compare the best agent of Part III with the contextual agents on
    the same bays (play() refuses an Agent on a ContextualBay).
    """

    def __init__(self, agent, n_spots=5, seed=None):
        super().__init__(n_spots, seed)
        self.agent = agent

    def reset(self):
        self.agent.reset()

    def choose(self, x):
        return self.agent.choose()

    def update(self, x, spot, catch):
        self.agent.update(spot, catch)


class LinearRegressionAgent(ContextualAgent):
    """Predicts the catch of each spot with a linear regression on x, then picks the best.

    Memory, for each spot k (ridge regression, updated online):
        A_k = lam * I + sum of phi phi^T     over the days spot k was fished
        b_k = sum of catch * phi
    with phi = _features(x). The prediction is phi^T theta_k, theta_k = A_k^-1 b_k.
    Linear regression fits this simulator well: the hidden average catch of a
    spot is linear in the scaled conditions (clipped at 0).

    Choice:
    1. Warm-up: until every spot has been fished `warmup` times, fish the
       least tried spot (a regression needs data).
    2. Then, with probability `epsilon` a random spot, otherwise the spot with
       the largest predicted catch for today's conditions.

    epsilon = 0 is "acting only on the predictions" (the failing case of
    Task 6); epsilon > 0 is the epsilon-greedy fix (Section 2.2).

    Parameters
    ----------
    lam : float
        Ridge penalty: keeps theta defined with fewer than 4 samples.
    warmup : int
        Samples of each spot collected before trusting the predictions.
    epsilon : float
        Exploration probability after the warm-up.
    """

    def __init__(self, n_spots=5, seed=None, lam=1.0, warmup=1, epsilon=0.0):
        super().__init__(n_spots, seed)
        self.lam, self.warmup, self.epsilon = lam, warmup, epsilon

    def reset(self):
        d = 4                                            # 1 + 3 features
        self.A = np.array([self.lam * np.eye(d) for _ in range(self.n_spots)])
        self.b = np.zeros((self.n_spots, d))
        self.N = np.zeros(self.n_spots)

    def predict(self, x):
        """Predicted catch of every spot for conditions x (array of length n_spots)."""
        phi = _features(x)
        return np.array([phi @ np.linalg.solve(self.A[k], self.b[k]) for k in range(self.n_spots)])

    def choose(self, x):
        if self.N.min() < self.warmup:                   # 1. warm-up
            return _argmax_random_tie(-self.N, self.rng)
        if self.rng.random() < self.epsilon:             # 2a. explore
            return int(self.rng.integers(self.n_spots))
        return _argmax_random_tie(self.predict(x), self.rng)  # 2b. best prediction

    def update(self, x, spot, catch):
        phi = _features(x)
        self.A[spot] += np.outer(phi, phi)
        self.b[spot] += catch * phi
        self.N[spot] += 1


class LinUCBAgent(LinearRegressionAgent):
    """Fix for "acting only on predictions": UCB (Section 2.6) on the linear regression.

    Same model as LinearRegressionAgent, but the score of spot k is
        phi^T theta_k + c * sqrt(phi^T A_k^-1 phi)
    The second term is the uncertainty of the prediction for TODAY's
    conditions: it is large for a spot rarely fished, or never fished in
    conditions like today's. It plays the role of sqrt(ln t / N(a)) in eq. 2.8,
    so a spot that looked bad after a few unlucky days is still re-tried, and
    the exploration is aimed at the conditions where we know little.
    (This is the LinUCB algorithm of Li et al., 2010.)

    Parameters
    ----------
    c : float
        Exploration strength (kg per unit of prediction standard error).
        c = 5 was the best of {1, 2.5, 5} on held-out tuning bays.
    """

    def __init__(self, n_spots=5, seed=None, lam=1.0, c=5.0):
        super().__init__(n_spots, seed, lam=lam, warmup=1, epsilon=0.0)
        self.c = c

    def choose(self, x):
        if self.N.min() < self.warmup:
            return _argmax_random_tie(-self.N, self.rng)
        phi = _features(x)
        scores = np.empty(self.n_spots)
        for k in range(self.n_spots):
            A_inv_phi = np.linalg.solve(self.A[k], phi)
            theta = np.linalg.solve(self.A[k], self.b[k])
            scores[k] = phi @ theta + self.c * np.sqrt(phi @ A_inv_phi)
        return _argmax_random_tie(scores, self.rng)


class KNNAgent(ContextualAgent):
    """Predicts the catch of each spot with K-nearest neighbours on x.

    Memory: for each spot, the list of (scaled conditions, catch) of the days
    it was fished.

    Prediction for spot k and today's conditions: average catch of the `k`
    past days at spot k whose scaled conditions are closest (Euclidean
    distance) to today's. With fewer than `k` days, the average of all of them.
    Unlike the linear regression, KNN assumes nothing about the shape of the
    relation, but needs more data and predicts badly in rarely seen conditions.

    Choice: warm-up (`warmup` days per spot), then epsilon-greedy on the
    predictions, as LinearRegressionAgent.
    """

    def __init__(self, n_spots=5, seed=None, k=10, warmup=3, epsilon=0.0):
        super().__init__(n_spots, seed)
        self.k, self.warmup, self.epsilon = k, warmup, epsilon

    def reset(self):
        self.X = [[] for _ in range(self.n_spots)]   # scaled conditions per spot
        self.y = [[] for _ in range(self.n_spots)]   # catches per spot

    def predict(self, x):
        z = np.asarray(x, dtype=float) / _X_SCALE
        pred = np.empty(self.n_spots)
        for s in range(self.n_spots):
            X, y = np.array(self.X[s]), np.array(self.y[s])
            nearest = np.argsort(np.linalg.norm(X - z, axis=1))[: self.k]
            pred[s] = y[nearest].mean()
        return pred

    def choose(self, x):
        counts = np.array([len(v) for v in self.y])
        if counts.min() < self.warmup:
            return _argmax_random_tie(-counts, self.rng)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_spots))
        return _argmax_random_tie(self.predict(x), self.rng)

    def update(self, x, spot, catch):
        self.X[spot].append(np.asarray(x, dtype=float) / _X_SCALE)
        self.y[spot].append(catch)
