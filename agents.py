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


def argmax_random_tie(values, rng):
    """Index of the largest value, ties broken uniformly at random with rng.

    np.argmax always returns the first maximum, which would bias every agent
    towards spot 0 whenever estimates are equal (for example at the start).
    """
    values = np.asarray(values, dtype=float)
    return int(rng.choice(np.flatnonzero(values == values.max())))


def check_probability(name, value):
    """Raise ValueError unless value is a number between 0 and 1 inclusive."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def check_step_size(alpha):
    """Raise ValueError unless alpha is None (sample average) or in (0, 1]."""
    if alpha is not None and not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be None or in (0, 1], got {alpha}")


def incremental_update(q, n, spot, catch, alpha):
    """Count a visit of spot and move its estimate q[spot] towards catch, in place.

    The step is 1 / n[spot] (sample average, Section 2.4) when alpha is None,
    otherwise the constant alpha (Section 2.5).
    """
    n[spot] += 1
    step = 1.0 / n[spot] if alpha is None else alpha
    q[spot] += step * (catch - q[spot])


class ThresholdRuleAgent(Agent):
    """Soulaimen's strategy, from his strategy notes in the log "bouazza".

    The notes, made precise. "Days since visited" of a spot is the number of
    days fished since its last visit (a spot never visited counts from day 0).

    1. Day 1: fish at start_spot, Seagrass Bed (Season 1 note: "in the middle").
    2. Stay while yesterday's catch is at least leave_below (Season 3 note:
       "When I got less than 6kg I changed the spot and lasted in one spot when
       it gave me more than 7kg"; this note leaves a spot only below 6 kg, so a
       catch between 6 and 7 kg also keeps it. The Season 1 note says about
       6.5 kg).
    3. On a day it leaves, if another spot has not been visited for
       forgotten_after days or more, test that spot, the longest forgotten one
       first (Season 3 note: "spent much time not visiting one spot (for 8
       days for example) and that I had to change the spot I'm actually in I
       test this forgotten spot").
    4. Otherwise move to the other spot with the best average catch so far. If
       no other spot has been visited yet, move to the least recently visited
       spot instead.

    Ties are broken at random with self.rng.

    Memory: total catch and visits per spot, the day of each spot's last
    visit, the current spot and yesterday's catch.
    Parameters: start_spot, leave_below (kg), forgotten_after (days).
    """

    def __init__(self, start_spot=2, leave_below=6.0, forgotten_after=8,
                 n_spots=5, seed=None):
        """Store the start spot, the leaving threshold (kg) and the day count.

        Raises ValueError for a start spot out of range or a day count below 1.
        """
        super().__init__(n_spots, seed)
        if not 0 <= start_spot < n_spots:
            raise ValueError(f"start_spot must be between 0 and {n_spots - 1}")
        if forgotten_after < 1:
            raise ValueError("forgotten_after must be at least 1")
        self.start_spot = start_spot
        self.leave_below = leave_below
        self.forgotten_after = forgotten_after

    def reset(self):
        """Forget every catch and visit of the previous season."""
        self.day = 0
        self.sums = np.zeros(self.n_spots)
        self.counts = np.zeros(self.n_spots)
        self.last_visit = np.zeros(self.n_spots)
        self.current = None
        self.last_catch = None

    def _days_since_other_visits(self):
        """Days fished since each spot's last visit, -inf for the current spot.

        A spot never visited counts from day 0.
        """
        since = self.day - self.last_visit
        since[self.current] = -np.inf
        return since

    def choose(self):
        """Apply the rules in the order of the class docstring."""
        if self.day == 0:
            return self.start_spot
        if self.last_catch >= self.leave_below:
            return self.current
        since = self._days_since_other_visits()
        visited = self.counts > 0
        visited[self.current] = False
        if since.max() >= self.forgotten_after or not visited.any():
            return argmax_random_tie(since, self.rng)
        averages = np.where(visited, self.sums / np.maximum(self.counts, 1), -np.inf)
        return argmax_random_tie(averages, self.rng)

    def update(self, spot, catch):
        """Record the catch and the day of this visit."""
        self.day += 1
        self.current = spot
        self.sums[spot] += catch
        self.counts[spot] += 1
        self.last_visit[spot] = self.day
        self.last_catch = catch


class ExploreThenCommitAgent(Agent):
    """Shubham's strategy, from his Season 1 and Season 2 notes in the log "kumarsh".

    "I visited each location and identified the one that yielded the most
    fish. Then, I tried to fish at the same spot multiple times." As a rule:
    while some spot has fewer than explore_rounds visits, fish at the spot
    with the fewest visits (the lowest spot number among equals, so a fresh
    season goes 0, 1, 2, 3, 4). After that, every day fish at the spot with
    the highest average catch so far, ties broken at random with self.rng.

    Memory: total catch and number of visits of each spot.
    Parameter: explore_rounds, how many times each spot is tried first.
    """

    def __init__(self, explore_rounds=1, n_spots=5, seed=None):
        """Store explore_rounds.

        Raises ValueError if explore_rounds is below 1.
        """
        super().__init__(n_spots, seed)
        if explore_rounds < 1:
            raise ValueError("explore_rounds must be at least 1")
        self.explore_rounds = explore_rounds

    def reset(self):
        """Forget the catches of the previous season."""
        self.sums = np.zeros(self.n_spots)
        self.counts = np.zeros(self.n_spots)

    def choose(self):
        """The least visited spot during exploration, then the best average."""
        if self.counts.min() < self.explore_rounds:
            return int(np.argmin(self.counts))
        return argmax_random_tie(self.sums / self.counts, self.rng)

    def update(self, spot, catch):
        """Add the catch to the spot's total."""
        self.sums[spot] += catch
        self.counts[spot] += 1


class SweepAgent(Agent):
    """Adhitya's strategy, from his Season 2 note in the log "kanakara" ("in a ordered way").

    The Season 2 log is two sweeps: each block of 5 days visits every spot
    once. As a rule: each day, fish at a spot not yet visited in the current
    sweep, chosen uniformly at random with self.rng. Once every spot has been
    visited, a new sweep starts. Catches are never used, so in expectation the
    rule is as good as random choice.

    Memory: the spots not yet visited in the current sweep. No parameters.
    """

    def reset(self):
        """Start a new sweep with every spot still to visit."""
        self.remaining = set(range(self.n_spots))

    def choose(self):
        """A random spot among those not yet visited in this sweep."""
        return int(self.rng.choice(sorted(self.remaining)))

    def update(self, spot, catch):
        """Mark the spot as visited, and start a new sweep once all are visited."""
        self.remaining.discard(spot)
        if not self.remaining:
            self.reset()


class HumanRuleAgent(ExploreThenCommitAgent):
    """Task 4: the best human strategy identified in Part II, written as a rule.

    The rule (Shubham's notes): fish once at each spot, in the order Old Pier,
    Rocky Point, Seagrass Bed, Deep Channel, Lighthouse Reef. From day 6 on,
    every day fish at the spot whose catches so far have the highest average,
    choosing at random between spots with exactly the same average.

    It is ExploreThenCommitAgent with explore_rounds=1. Of the team's rules,
    it has the lowest mean regret summed over the three seasons on 1000
    random bays per season (experiment.ipynb, Part II). ThresholdRuleAgent is
    better in Season 3 only.

    Memory: total catch and number of visits of each spot. No parameters.
    """

    def __init__(self, n_spots=5, seed=None):
        """One exploration round, as in the notes."""
        super().__init__(explore_rounds=1, n_spots=n_spots, seed=seed)


# Task 4: add below at least two agents implementing methods from the reading
# (Sutton and Barto, Chapter 2), as subclasses of Agent. Document each class:
# what it keeps in memory, how it chooses, what its parameters mean.


class EpsilonGreedyAgent(Agent):
    """Epsilon-greedy action-value method (Sutton and Barto, Sections 2.2 to 2.6).

    Memory: an estimate Q of the average catch of each spot and the number of
    visits N of each spot.

    Choice: with probability epsilon a spot chosen uniformly at random,
    otherwise the spot with the highest Q (ties broken at random).

    Update after a catch R at spot s: Q[s] += step * (R - Q[s]), where step is
    1 / N[s] (sample average, Section 2.4) when alpha is None, or the constant
    alpha (Section 2.5), which weights recent catches more and so can follow
    the change of Season 3.

    Parameters:
    - epsilon: exploration probability, between 0 and 1.
    - alpha: None for sample averages, or a constant step size in (0, 1].
    - initial_value: starting Q of every spot. A value above any possible
      average (for example 15 kg) with epsilon 0 and a constant alpha is the
      optimistic initial values method (Section 2.6). With sample averages the
      first visit replaces the initial value, so optimism needs alpha.
    """

    def __init__(self, epsilon=0.1, alpha=None, initial_value=0.0, n_spots=5, seed=None):
        """Store the parameters.

        Raises ValueError if epsilon or alpha is out of range.
        """
        super().__init__(n_spots, seed)
        check_probability("epsilon", epsilon)
        check_step_size(alpha)
        self.epsilon = epsilon
        self.alpha = alpha
        self.initial_value = initial_value

    def reset(self):
        """Set every estimate back to initial_value and every count to 0."""
        self.q = np.full(self.n_spots, float(self.initial_value))
        self.n = np.zeros(self.n_spots)

    def choose(self):
        """Explore with probability epsilon, otherwise exploit the best estimate."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_spots))
        return argmax_random_tie(self.q, self.rng)

    def update(self, spot, catch):
        """Move the estimate of spot towards the catch."""
        incremental_update(self.q, self.n, spot, catch, self.alpha)


class UCBAgent(Agent):
    """Upper-confidence-bound action selection (Sutton and Barto, Section 2.7).

    Memory: an estimate Q and a visit count N for each spot, and the day t.

    Choice: a spot never visited first (chosen at random among them). Once
    every spot has been visited, the spot with the highest
    Q[s] + c * sqrt(ln(t) / N[s]), where t is today's day number (1, 2, ...).
    The bonus is large for spots visited rarely, so they get re-tested.

    Update: as in EpsilonGreedyAgent, a sample average (alpha None) or a
    constant step size alpha.

    Parameters:
    - c: exploration strength in kg, at least 0. Catches have a noise
      standard deviation of 2.5 kg, so useful values are of that order.
    - alpha: None for sample averages, or a constant step size in (0, 1].
    """

    def __init__(self, c=2.0, alpha=None, n_spots=5, seed=None):
        """Store the parameters.

        Raises ValueError if c is negative or alpha is out of range.
        """
        super().__init__(n_spots, seed)
        if c < 0:
            raise ValueError(f"c must be at least 0, got {c}")
        check_step_size(alpha)
        self.c = c
        self.alpha = alpha

    def reset(self):
        """Forget the estimates, the counts and the day."""
        self.q = np.zeros(self.n_spots)
        self.n = np.zeros(self.n_spots)
        self.t = 0

    def choose(self):
        """Visit untried spots first, then maximise the upper confidence bound."""
        untried = self.n == 0
        if untried.any():
            return argmax_random_tie(untried, self.rng)
        day = self.t + 1
        bound = self.q + self.c * np.sqrt(np.log(day) / self.n)
        return argmax_random_tie(bound, self.rng)

    def update(self, spot, catch):
        """Move the estimate of spot towards the catch and count the day."""
        self.t += 1
        incremental_update(self.q, self.n, spot, catch, self.alpha)


# Task 6 (optional): add your contextual agents below, as subclasses of
# ContextualAgent.


class LinearRegressionAgent(ContextualAgent):
    """Predicts each spot's catch from the conditions with linear regression.

    Model: for each spot s, catch ~ theta_s . z with z = [1, wind / 40,
    waves / 3, tide / 2]. The divisors are feature_scale, the maximum of each
    condition stated in the fishing.py notes, so all features lie in [0, 1].

    Memory: for each spot, A_s = ridge * I + sum of z z^T and b_s = sum of
    z * catch over the days fished there. theta_s = A_s^-1 b_s is the ridge
    least-squares fit. A spot with no data has theta_s = 0, so its prediction
    is 0 kg.

    Choice: with probability epsilon a spot chosen uniformly at random.
    Otherwise the spot with the highest score theta_s . z + bonus * width_s,
    where width_s = sqrt(z^T A_s^-1 z) is large when spot s has few
    observations in conditions like today's (ties broken at random).

    With epsilon 0 and bonus 0 the agent acts only on its predictions. On
    day 1 every prediction is 0, so it picks a random spot. That spot then
    predicts a positive catch while unvisited spots stay at 0, so the agent
    can lock onto it for the whole season. Two fixes:
    - epsilon > 0: random days keep every spot sampled (Section 2.2 idea).
    - bonus > 0: an upper confidence bound on the prediction, the
      contextual form of UCB (Section 2.7), known as LinUCB. Spots and
      conditions that were rarely observed get tested, others do not.

    Parameters:
    - epsilon: exploration probability, between 0 and 1.
    - bonus: weight of the confidence width, at least 0.
    - ridge: regularisation strength, above 0. Keeps A_s invertible before a
      spot has 4 observations.
    - feature_scale: the divisor of each condition.
    """

    def __init__(self, epsilon=0.0, bonus=0.0, ridge=1.0,
                 feature_scale=(40.0, 3.0, 2.0), n_spots=5, seed=None):
        """Store the parameters.

        Raises ValueError if epsilon is out of range, bonus is negative,
        ridge is not positive, or feature_scale does not hold 3 positive numbers.
        """
        super().__init__(n_spots, seed)
        check_probability("epsilon", epsilon)
        if bonus < 0:
            raise ValueError(f"bonus must be at least 0, got {bonus}")
        if ridge <= 0:
            raise ValueError(f"ridge must be above 0, got {ridge}")
        scale = np.asarray(feature_scale, dtype=float)
        if scale.shape != (3,) or np.any(scale <= 0):
            raise ValueError("feature_scale must hold 3 positive numbers")
        self.epsilon = epsilon
        self.bonus = bonus
        self.ridge = ridge
        self.feature_scale = scale

    def reset(self):
        """Forget every observation: A_s back to ridge * I and b_s to 0."""
        dim = len(self.feature_scale) + 1
        self.a = np.stack([self.ridge * np.eye(dim) for _ in range(self.n_spots)])
        self.b = np.zeros((self.n_spots, dim))

    def _features(self, x):
        """The feature vector z = [1, x / feature_scale] of conditions x."""
        return np.concatenate(([1.0], np.asarray(x, dtype=float) / self.feature_scale))

    def predict(self, x):
        """Predicted catch of every spot for conditions x (array of n_spots)."""
        z = self._features(x)
        theta = np.linalg.solve(self.a, self.b[:, :, None])[:, :, 0]
        return theta @ z

    def confidence_width(self, x):
        """sqrt(z^T A_s^-1 z) of every spot for conditions x (array of n_spots)."""
        z = self._features(x)
        a_inv_z = np.linalg.solve(self.a, np.tile(z, (self.n_spots, 1))[:, :, None])[:, :, 0]
        return np.sqrt(a_inv_z @ z)

    def choose(self, x):
        """Explore with probability epsilon, otherwise the best score."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_spots))
        score = self.predict(x)
        if self.bonus > 0:
            score = score + self.bonus * self.confidence_width(x)
        return argmax_random_tie(score, self.rng)

    def update(self, x, spot, catch):
        """Add the observation (x, catch) to the regression of spot."""
        z = self._features(x)
        self.a[spot] += np.outer(z, z)
        self.b[spot] += catch * z


class ContextFreeAgent(ContextualAgent):
    """Plays a Part III agent on the contextual bay, ignoring the conditions.

    Used to compare the contextual agents with the best agent of Part III.
    Memory and choices are those of the wrapped agent, which never sees x.

    Parameter: agent, an instance of a subclass of Agent.
    """

    def __init__(self, agent):
        """Wrap agent. Raises TypeError if it is not an Agent."""
        if not isinstance(agent, Agent):
            raise TypeError("agent must be an instance of Agent")
        super().__init__(agent.n_spots)
        self.agent = agent

    def reset(self):
        """Reset the wrapped agent."""
        self.agent.reset()

    def choose(self, x):
        """The wrapped agent's choice. The conditions x are ignored."""
        return self.agent.choose()

    def update(self, x, spot, catch):
        """Pass the spot and catch to the wrapped agent. x is ignored."""
        self.agent.update(spot, catch)
