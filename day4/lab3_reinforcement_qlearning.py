"""
LAB 3 - REINFORCEMENT LEARNING: Tabular Q-Learning From Scratch
================================================================
ML Masterclass · Module 2 companion lab

Goal: make the agent-environment loop tangible. No frameworks — the
whole algorithm is ~40 lines. An agent learns to cross a gridworld with
a cliff, purely from reward signals, using epsilon-greedy exploration
and the Bellman update.

The grid (4 rows x 6 cols):
      S = start   G = goal (+50)   C = cliff (-100, reset)   step = -1

      . . . . . .
      . . . . . .
      . . . . . .
      S C C C C G

Run:  python lab3_reinforcement_qlearning.py
Deps: numpy only.   Time: < 10 seconds.
"""

import numpy as np

RNG = np.random.default_rng(3)

# ----------------------------------------------------------------------
# SECTION 1 — The environment (the "world" the agent samples)
# ----------------------------------------------------------------------
ROWS, COLS = 4, 6
START, GOAL = (3, 0), (3, 5)
CLIFF = {(3, c) for c in range(1, 5)}
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # U D L R
ARROWS  = {0: "^", 1: "v", 2: "<", 3: ">"}

def step(state, action):
    """Environment dynamics: returns (next_state, reward, done)."""
    dr, dc = ACTIONS[action]
    r = min(max(state[0] + dr, 0), ROWS - 1)
    c = min(max(state[1] + dc, 0), COLS - 1)
    nxt = (r, c)
    if nxt in CLIFF:
        return START, -100.0, False        # fall: big penalty, back to start
    if nxt == GOAL:
        return nxt, +50.0, True
    return nxt, -1.0, False                # living cost: encourages short paths

# ----------------------------------------------------------------------
# SECTION 2 — The agent: a Q-table and the Bellman update
# ----------------------------------------------------------------------
# Q[s,a] estimates "total future reward if I take action a in state s
# and act well afterwards". Learning rule:
#   Q[s,a] += alpha * (r + gamma * max_a' Q[s',a'] - Q[s,a])
#               (         target: reward + discounted best future        )
# ----------------------------------------------------------------------
ALPHA, GAMMA = 0.1, 0.95          # learning rate, discount factor
EPS_START, EPS_END = 1.0, 0.05    # exploration schedule
EPISODES = 600

Q = np.zeros((ROWS, COLS, 4))

def policy(state, eps):
    """Epsilon-greedy: explore with prob eps, else exploit best known."""
    if RNG.uniform() < eps:
        return RNG.integers(4)                      # explore
    return int(np.argmax(Q[state[0], state[1]]))    # exploit

returns = []
for ep in range(EPISODES):
    eps = max(EPS_END, EPS_START * (1 - ep / 400))  # decay exploration
    s, total, done, steps = START, 0.0, False, 0
    while not done and steps < 200:
        a = policy(s, eps)
        s2, r, done = step(s, a)
        # ----- the entire learning algorithm is this one line: -----
        Q[s[0], s[1], a] += ALPHA * (r + GAMMA * Q[s2[0], s2[1]].max()
                                     - Q[s[0], s[1], a])
        s, total, steps = s2, total + r, steps + 1
    returns.append(total)

# ----------------------------------------------------------------------
# SECTION 3 — Did it learn? Watch the return curve climb
# ----------------------------------------------------------------------
print("=" * 60)
print("SECTION 3 · Learning curve (mean return per 100 episodes)")
print("=" * 60)
for i in range(0, EPISODES, 100):
    chunk = returns[i:i + 100]
    bar = "#" * max(0, int((np.mean(chunk) + 120) / 6))
    print(f"episodes {i:>3}-{i+99:<3}  avg return {np.mean(chunk):>8.1f}  {bar}")

# ----------------------------------------------------------------------
# SECTION 4 — The learned policy, visualized
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("SECTION 4 · Greedy policy (what the agent now believes)")
print("=" * 60)
for r in range(ROWS):
    row = ""
    for c in range(COLS):
        if (r, c) == GOAL:    row += "  G "
        elif (r, c) in CLIFF: row += "  C "
        elif (r, c) == START: row += " S" + ARROWS[int(np.argmax(Q[r, c]))] + " "
        else:                 row += "  " + ARROWS[int(np.argmax(Q[r, c]))] + " "
    print(row)

print("""
Expected result: the agent climbs OUT of the start cell, runs along a
safe row, and descends into the goal — it learned to respect the cliff
from reward alone. Nobody coded 'avoid the cliff'.

KEY IDEAS DEMONSTRATED
  - Exploration vs exploitation: eps decays 1.0 -> 0.05; early chaos
    funds later competence.
  - Credit assignment: the +50 at the goal propagates BACKWARD through
    the Q-table, one Bellman update at a time.
  - Reward design IS the spec: change the step cost from -1 to 0 and
    the agent no longer cares about path length. Reward mis-specification
    -> confident wrong behaviour (same failure class as RLHF reward
    hacking in LLMs, Module 6).

DISCUSSION
  1. Set GAMMA=0.5 (myopic agent). How does the path change and why?
  2. Set EPS_END=0.0 from the start (no exploration). What happens?
  3. This Q-table has 96 cells. A chess Q-table would need ~10^43 rows —
     hence DEEP RL: replace the table with a neural network (DQN).
""")
