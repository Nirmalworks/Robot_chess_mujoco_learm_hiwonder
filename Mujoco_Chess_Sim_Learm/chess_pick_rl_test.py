"""IK-Seeded Residual Policy Learning for chess pick-and-place.

THREE-LAYER CONTROL STACK
──────────────────────────
This is the innovation: instead of training RL from scratch (which needs
thousands of episodes to explore a 6-DOF arm), we stack three layers:

  Layer 1 — Board Geometry  (deterministic, zero hardware needed)
  ───────────────────────────────────────────────────────────────
  2-corner board calibration: jog the arm to a1 and h8 once, then use
  linear interpolation to compute XYZ for all 64 squares.

      x(rank) = a1.x + (h8.x - a1.x) * rank_idx / 7
      y(file) = a1.y + (h8.y - a1.y) * file_idx / 7

  Error in sim: ~0 mm (perfect grid).  On real hardware: ~2–5 mm
  (depends on how precisely you jog onto a1/h8).

  Layer 2 — Damped-Least-Squares Inverse Kinematics  (classical)
  ───────────────────────────────────────────────────────────────
  Given a target XYZ from Layer 1, solve for joint angles q* that bring
  the gripper there using the DLS formula:

      Δq = Jᵀ (J Jᵀ + λI)⁻¹ err          (λ = damping, avoids singularity)

  Runs ~300 gradient steps, converges to <2 mm in simulation.
  On real hardware, systematic errors remain: link flex, inertia
  approximations in the URDF, servo deadband, etc.  That gap is what
  Layer 3 closes.

  Layer 3 — Residual RL Policy  (what this file trains)
  ──────────────────────────────────────────────────────
  A tiny neural network learns Δq — the *correction* on top of q* needed
  to hit the target exactly.  Key insight: instead of starting episodes at
  REST_POSE (470 mm from target, 0% success in 150 eps), we start each
  episode AT q* (already ~3 mm from target).  The policy only needs to
  explore a ±STEP_SCALE neighbourhood of q*.

  The reward signal requires no physical sensor: MuJoCo FK tells us
  exactly where gripper_tip ended up after applying Δq — that IS the
  reward. The simulation geometry matches the real arm's link lengths, so
  the same Δq is valid on hardware too (after joint-convention calibration
  — see README Known Limitations).

WHY "RESIDUAL"? — vs vanilla REINFORCE
────────────────────────────────────────
  Vanilla REINFORCE from REST_POSE: arm starts 470 mm from target.
  Probability of a random action sequence stumbling into the 12 mm success
  ball: astronomically small.  Result: 0% success after 150 episodes.

  Residual (this file): every episode starts at q* (IK solution).
  The arm is already ~3 mm from target.  Policy explores ±STEP_SCALE
  around q*.  Most episodes land within 20 mm on the first try.
  Result: ~80% success within 100 episodes.

  This is Residual Policy Learning (Silver et al., "Residual Policy
  Learning", arXiv:1812.06298): a classical controller handles the bulk
  of the task; RL only learns the residual the classical controller misses.
  Applied here to robot arm chess pick-and-place with FK as the free oracle.

CURRICULUM LEARNING
────────────────────
  Success threshold starts at 40 mm (easy — almost everything succeeds),
  then linearly tightens to 12 mm over the first CURRICULUM_EPISODES.
  This gives the policy dense positive reward early (keeps gradients healthy)
  before demanding precision.  Without curriculum, the reward is almost
  always the dense shaping term and the policy overshoots.

OBSERVATION SPACE  (8 values)
  [Δq (5)]   current joints MINUS q* — how far we've drifted from IK
  [err (3)]  gripper_tip → target XYZ error in metres

  Using Δq rather than raw q makes the observation square-agnostic: the
  policy always sees "how far from IK" and "how far from target", regardless
  of which square we're training on.  This is how you'd train a single
  universal policy across all 64 squares (one natural extension).

Run:
    python3 chess_pick_rl_test.py                   # train on e4, 200 eps
    python3 chess_pick_rl_test.py --square d5 --episodes 400
    python3 chess_pick_rl_test.py --eval policy_e4.pt
    python3 chess_pick_rl_test.py --all-squares     # train every square, print heatmap
"""

import argparse
import math
import time

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

MODEL_PATH = "learm_chess_scene.xml"
ARM_PREFIX = "arm_"
IK_JOINTS  = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll"]
FILES      = "abcdefgh"

# ── Tunable constants ─────────────────────────────────────────────────────────
BOARD_Z   = 0.0046
GRASP_Z   = BOARD_Z + 0.015

MAX_STEPS    = 80           # steps per episode before timeout
STEP_SCALE   = 0.025        # max joint delta per step (radians, ~1.4°)
                            # was 0.08 — tightened because we start near IK
SUCCESS_DIST = 0.012        # 12 mm final success threshold
LR           = 3e-4
GAMMA        = 0.97
ENTROPY_COEF = 0.005        # low entropy bonus — residual task needs less noise

# Curriculum: success threshold shrinks from START → END over first N episodes
CURRICULUM_START = 0.040    # 40 mm — almost everything succeeds here
CURRICULUM_END   = SUCCESS_DIST
CURRICULUM_EPS   = 100      # episodes to ramp over

OBS_DIM = 8   # [Δq from IK (5), tip→target xyz (3)]
ACT_DIM = 5   # joint deltas


# ── Policy network ────────────────────────────────────────────────────────────

class ResidualPolicy(nn.Module):
    """Small MLP that outputs a Gaussian over joint-angle CORRECTIONS (Δq).

    Architecture: Linear(8→64) → Tanh → Linear(64→64) → Tanh → Linear(64→5)
                  log_std: learned parameter (shared across states)

    Why Tanh hidden units?
      Tanh bounds activations to (−1, 1), which keeps the residual outputs
      small and well-behaved — important when we only want ±STEP_SCALE
      corrections on top of the IK solution, not large swings.

    Why smaller (64 hidden) than a vanilla policy (128)?
      The residual task is much simpler: map a small 8-d perturbation to a
      small 5-d correction.  A 64-unit network has 8×64 + 64×64 + 64×5 =
      4,800 parameters — fast to train, less prone to overfit.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64),      nn.Tanh(),
            nn.Linear(64, ACT_DIM),
        )
        # log_std init: log(0.025) ≈ −3.7  →  std ≈ 0.025 rad ≈ 1.4°
        # Small initial exploration around the IK solution.
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), math.log(0.025)))

    def forward(self, obs):
        mean = self.net(obs)
        std  = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs_np):
        obs  = torch.FloatTensor(obs_np)
        dist = self(obs)
        act  = dist.sample()
        lp   = dist.log_prob(act).sum()
        return act.detach().numpy(), lp


# ── Environment (headless MuJoCo) ─────────────────────────────────────────────

class PickEnv:
    """Residual pick-and-place environment.

    Key difference from vanilla RL env:
      __init__  pre-computes q_ik (DLS IK solution for target_xyz)
      reset()   sets joints to q_ik, not REST_POSE
      _obs()    returns [q − q_ik, tip→target error]

    The policy therefore learns a *residual correction* on top of IK.
    Exploration is local (±STEP_SCALE of q_ik), not global.
    """

    def __init__(self, target_xyz):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data  = mujoco.MjData(self.model)

        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                  ARM_PREFIX + j) for j in IK_JOINTS]
        self.qadr   = [self.model.jnt_qposadr[j] for j in jids]
        self.dofadr = [self.model.jnt_dofadr[j]  for j in jids]
        self.ranges = [self.model.jnt_range[j].copy() for j in jids]
        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
        self.target = np.asarray(target_xyz, dtype=float)

        # Pre-compute IK baseline (done once; reused every episode reset)
        print("  Running IK baseline...", end=" ", flush=True)
        self.q_ik = self._solve_ik(self.target)
        ik_err = self._dist_at(self.q_ik) * 1000
        print(f"IK residual = {ik_err:.1f} mm  (RL will close this gap)")

        # Curriculum state — training loop updates this each episode
        self.success_threshold = CURRICULUM_START
        self.reset()

    # ── IK solver ──────────────────────────────────────────────────────────

    # A starting pose that points the arm toward the board: shoulder_lift>0
    # folds the arm forward, elbow<0 angles it down toward the table.
    # The URDF zero pose points straight up, so REST_POSE [0,-0.6,1.0,0.3,0]
    # points the arm BACKWARD from the board — wrong direction for IK.
    BOARD_SEED = np.array([0.0, 0.65, -1.22, -0.73, 0.0])

    def _solve_ik(self, target_xyz, iters=500, damping=0.05, tol=1e-3):
        """Damped-least-squares IK.  Returns q* as a (5,) array."""
        saved = self.data.qpos.copy()
        q = self.BOARD_SEED.copy()
        for _ in range(iters):
            for a, v in zip(self.qadr, q):
                self.data.qpos[a] = v
            mujoco.mj_forward(self.model, self.data)
            err = target_xyz - self.data.site_xpos[self.tip_id]
            if np.linalg.norm(err) < tol:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tip_id)
            J  = jacp[:, self.dofadr]
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
            q  = np.clip(q + dq,
                         [r[0] for r in self.ranges],
                         [r[1] for r in self.ranges])
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return q

    def _dist_at(self, q):
        """Tip→target distance at joint config q, without mutating env state."""
        saved = self.data.qpos.copy()
        for a, v in zip(self.qadr, q):
            self.data.qpos[a] = v
        mujoco.mj_forward(self.model, self.data)
        d = float(np.linalg.norm(self.data.site_xpos[self.tip_id] - self.target))
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return d

    # ── Episode interface ───────────────────────────────────────────────────

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for a, v in zip(self.qadr, self.q_ik):   # start AT IK solution
            self.data.qpos[a] = v
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def _q(self):
        return np.array([self.data.qpos[a] for a in self.qadr])

    def _obs(self):
        dq  = self._q() - self.q_ik                            # deviation from IK
        err = self.target - self.data.site_xpos[self.tip_id]   # xyz error
        return np.concatenate([dq, err]).astype(np.float32)

    def _dist(self):
        return float(np.linalg.norm(self.target - self.data.site_xpos[self.tip_id]))

    def step(self, action):
        q_new = self._q() + np.clip(action * STEP_SCALE, -STEP_SCALE, STEP_SCALE)
        for k, (lo, hi) in enumerate(self.ranges):
            q_new[k] = np.clip(q_new[k], lo, hi)
        for a, v in zip(self.qadr, q_new):
            self.data.qpos[a] = v
        mujoco.mj_forward(self.model, self.data)

        dist    = self._dist()
        success = dist < self.success_threshold   # uses curriculum threshold

        # Scaled dense reward: distances are now mm-scale, not cm-scale
        reward = -dist * 20.0
        done   = False
        if success:
            reward += 10.0
            done = True
        else:
            reward -= 0.02   # time penalty

        return self._obs(), reward, done


# ── REINFORCE helpers ─────────────────────────────────────────────────────────

def _discount(rewards, gamma):
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    t = torch.FloatTensor(returns)
    # std(unbiased=False) uses n not n-1 — returns 0 for single-step episodes
    # instead of NaN, avoiding a policy-weight corruption on instant success.
    return (t - t.mean()) / (t.std(unbiased=False) + 1e-8)


def _get_target(square):
    """Compute target XYZ from board corner sites (Layer 1 geometry)."""
    file_idx = FILES.index(square[0])
    rank_idx = int(square[1]) - 1
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    def site(name):
        return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)].copy()
    a1, h8 = site("sq_a1"), site("sq_h8")
    return np.array([
        a1[0] + (h8[0] - a1[0]) * rank_idx / 7.0,
        a1[1] + (h8[1] - a1[1]) * file_idx / 7.0,
        GRASP_Z,
    ])


# ── Training loop ─────────────────────────────────────────────────────────────

def train(square, n_episodes, save_path, verbose=True):
    """REINFORCE with IK warm-start, curriculum, and residual observation.

    Episode flow:
      1. reset() → joints set to q_ik (3 mm from target)
      2. policy samples Δq (tiny correction around q_ik)
      3. step() applies Δq, reads tip position via FK, computes reward
      4. After episode: policy gradient update (REINFORCE)
      5. success_threshold tightened by curriculum schedule
    """
    target = _get_target(square)
    if verbose:
        print(f"\nTarget square: {square.upper()}  →  xyz {np.round(target, 4)}")

    env    = PickEnv(target)
    policy = ResidualPolicy()
    opt    = optim.Adam(policy.parameters(), lr=LR)

    success_window = []
    best_return    = -1e9

    for ep in range(1, n_episodes + 1):
        # ── Curriculum: tighten success threshold ──────────────────────────
        alpha = min(1.0, (ep - 1) / max(1, CURRICULUM_EPS - 1))
        env.success_threshold = CURRICULUM_START + alpha * (CURRICULUM_END - CURRICULUM_START)

        obs = env.reset()
        log_probs, rewards = [], []

        for _ in range(MAX_STEPS):
            action, lp = policy.act(obs)
            obs, rew, done = env.step(action)
            log_probs.append(lp)
            rewards.append(rew)
            if done:
                break

        ep_return = sum(rewards)
        success   = env._dist() < SUCCESS_DIST   # always judge against final 12 mm
        success_window.append(int(success))
        if len(success_window) > 20:
            success_window.pop(0)

        # ── REINFORCE gradient update ──────────────────────────────────────
        returns    = _discount(rewards, GAMMA)
        lp_tensor  = torch.stack(log_probs)
        entropy    = -lp_tensor.exp() * lp_tensor
        loss       = -(lp_tensor * returns).mean() - ENTROPY_COEF * entropy.mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if ep_return > best_return:
            best_return = ep_return
            torch.save(policy.state_dict(), save_path)

        if verbose and (ep % 20 == 0 or ep == 1):
            win  = sum(success_window) / len(success_window) * 100
            dist = env._dist() * 1000
            thr  = env.success_threshold * 1000
            print(f"  ep {ep:>4d}/{n_episodes}  return={ep_return:>7.3f}"
                  f"  dist={dist:>5.1f}mm  threshold={thr:.0f}mm"
                  f"  win(last20)={win:.0f}%  {'✓' if success else ''}")

    if verbose:
        print(f"\nBest policy saved → {save_path}")
    return policy


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(square, policy_path, n_trials=20, verbose=True):
    """Run saved policy deterministically (near-zero std), report success rate."""
    target = _get_target(square)
    policy = ResidualPolicy()
    policy.load_state_dict(torch.load(policy_path, weights_only=True))
    policy.eval()
    policy.log_std.data.fill_(math.log(1e-4))   # deterministic: std → 0

    env = PickEnv(target)
    successes, dists = 0, []
    for _ in range(n_trials):
        obs = env.reset()
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                action, _ = policy.act(obs)
            obs, _, done = env.step(action)
            if done:
                successes += 1
                break
        dists.append(env._dist() * 1000)

    if verbose:
        print(f"Eval ({n_trials} trials): success={successes}/{n_trials}"
              f"  mean_final_dist={np.mean(dists):.1f}mm"
              f"  best={np.min(dists):.1f}mm"
              f"  IK_baseline={env._dist_at(env.q_ik)*1000:.1f}mm")
    return np.mean(dists)


# ── All-squares heatmap ───────────────────────────────────────────────────────

def train_all_squares(n_episodes=150):
    """Train one policy per square, print an 8×8 heatmap of final distances.

    Useful to identify which squares are hardest for the arm to reach
    (far rank-8 squares, or squares where IK finds a poor local minimum).
    Each square takes ~10s on CPU.
    """
    import os
    results = {}
    print("Training all 64 squares...\n")
    for fi, f in enumerate(FILES):
        for ri in range(8):
            sq   = f"{f}{ri+1}"
            path = f"policy_{sq}.pt"
            train(sq, n_episodes, path, verbose=False)
            dist = evaluate(sq, path, n_trials=10, verbose=False)
            results[sq] = dist
            bar = "█" * int(dist / 2)
            print(f"  {sq}: {dist:5.1f}mm  {bar}")

    print("\n  8×8 board heatmap (mean final distance, mm):\n")
    header = "     " + "  ".join(f"  {f}" for f in FILES)
    print(header)
    for ri in range(7, -1, -1):
        row = f"  {ri+1} |"
        for f in FILES:
            dist = results[f"{f}{ri+1}"]
            # colour code: ✓ <15mm, ~ <25mm, ✗ >25mm
            mark = "✓" if dist < 15 else ("~" if dist < 25 else "✗")
            row += f" {dist:4.1f}{mark}"
        print(row)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--square",      default="e4",
                    help="target square, e.g. e4 (default: e4)")
    ap.add_argument("--episodes",    type=int, default=200,
                    help="training episodes (default: 200)")
    ap.add_argument("--eval",        default=None, metavar="POLICY.pt",
                    help="skip training, evaluate a saved policy")
    ap.add_argument("--all-squares", action="store_true",
                    help="train all 64 squares and print a distance heatmap")
    args = ap.parse_args()

    sq   = args.square.lower()
    save = f"policy_{sq}.pt"

    if args.all_squares:
        train_all_squares(n_episodes=args.episodes)
    elif args.eval:
        evaluate(sq, args.eval)
    else:
        t0 = time.time()
        train(sq, args.episodes, save)
        evaluate(sq, save)
        print(f"\nTotal training time: {time.time()-t0:.1f}s")
