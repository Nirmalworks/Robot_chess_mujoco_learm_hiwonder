"""Sample: distance geometry + online self-learning pick-and-place.

What this file is for
---------------------
- Shows exactly what reference measurements you need to give the system.
- Demonstrates distance decomposition from arm base to any chess square.
- Implements a simple online learning loop that trains pick-and-place using
  only MuJoCo's own forward kinematics as the reward signal -- no external
  sensor, no camera, no human feedback needed.

The learning part is model-based: the robot tries a joint configuration,
we run forward kinematics (FK) to predict where the gripper actually ended up
(using our MuJoCo model as an internal 'oracle'), compute a reward from how
close that was to the target, then nudge the policy toward better configs.
On real hardware, the same joint command goes to the servo over serial --
the FK-computed position is still a valid prediction because the URDF geometry
matches the real arm (same joint offsets/link lengths).

Quick start
-----------
  python3 dev_pickup_sample.py           # runs entirely in simulation
  python3 dev_pickup_sample.py --serial  # also drives real arm over serial

To get A1_POS / H8_POS from your own arm:
  1. python3 view_sim.py           (manual jogging mode)
  2. Drag sliders until gripper_tip is over square a1, note the xyz in the
     bottom-left overlay, paste into A1_POS below.
  3. Do the same for h8 -> H8_POS.
"""

import argparse
import time

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# ONE-TIME MEASUREMENTS (fill these in from your physical/sim calibration)
# ---------------------------------------------------------------------------
ARM_BASE_XYZ = np.array([-0.03, 0.0, 0.0])   # ARM_MOUNT_POS from build_scene.py

# Paste your measured gripper_tip positions here (read from viewer overlay):
A1_POS = np.array([0.04, -0.07, 0.01])   # ← jog to a1, read xyz, paste here
H8_POS = np.array([0.18,  0.07, 0.01])   # ← jog to h8, read xyz, paste here

BOARD_SURFACE_Z = 0.0046   # from build_scene.py
HOVER_Z_OFFSET  = 0.06     # how far above surface to hover before descending
GRASP_Z_OFFSET  = 0.015    # gripper_tip height while holding a piece

MODEL_PATH = "learm_chess_scene.xml"
FILES = "abcdefgh"
ARM_PREFIX = "arm_"
IK_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll"]


# ---------------------------------------------------------------------------
# GEOMETRY: arm-base to square
# ---------------------------------------------------------------------------

def square_xyz(file_idx, rank_idx):
    """Interpolate a square's world position from A1_POS and H8_POS."""
    u = rank_idx / 7.0   # 0 at rank 1, 1 at rank 8
    v = file_idx / 7.0   # 0 at file a, 1 at file h
    x = A1_POS[0] + (H8_POS[0] - A1_POS[0]) * u
    y = A1_POS[1] + (H8_POS[1] - A1_POS[1]) * v
    z = (A1_POS[2] + H8_POS[2]) / 2.0
    return np.array([x, y, z])


def distance_breakdown(square):
    """
    From ARM_BASE_XYZ to `square`, break the distance into:
      - horizontal:  sqrt(Δx² + Δy²)  -- the actual reach the arm needs
      - Δx, Δy:      directional components in the table plane
      - Δz:          height difference (usually negative; arm dips down)

    If horizontal > max_reach, the arm can't get there.
    23 cm example: if the square is at (0.20, 0.0, 0.01) and base is at
    (-0.03, 0.0, 0.0), then Δx=0.23, Δy=0, horizontal=0.23 m = 23 cm.
    """
    delta = square - ARM_BASE_XYZ
    horizontal = float(np.linalg.norm(delta[:2]))
    return {
        "horizontal_m": round(horizontal, 4),
        "horizontal_cm": round(horizontal * 100, 2),
        "dx": round(float(delta[0]), 4),
        "dy": round(float(delta[1]), 4),
        "dz": round(float(delta[2]), 4),
        "within_reach": horizontal < 0.30,   # rough LeArm max reach
    }


def print_board_distances():
    """Print every square's distance from the arm base -- useful for setup."""
    print(f"{'Sq':4s} {'horiz_cm':>9s} {'dx_cm':>7s} {'dy_cm':>7s} {'reachable':>10s}")
    for fi in range(8):
        for ri in range(8):
            sq = f"{FILES[fi]}{ri+1}"
            xyz = square_xyz(fi, ri)
            d = distance_breakdown(xyz)
            reach = "YES" if d["within_reach"] else "NO !"
            print(f"{sq:4s} {d['horizontal_cm']:9.2f} {d['dx']*100:7.2f} {d['dy']*100:7.2f} {reach:>10s}")


# ---------------------------------------------------------------------------
# ONLINE SELF-LEARNING PICK-AND-PLACE
# ---------------------------------------------------------------------------
# Algorithm: Perturbation-gradient / zero-order policy search.
#
# We have an IK-computed baseline joint configuration (q_ik) for each target.
# The POLICY adds a learned per-square offset (δq) on top of that baseline.
# Each episode:
#   1. Sample action = q_ik + δq + noise
#   2. Run MuJoCo FK with that action → get gripper_tip position
#   3. Compute reward = -distance(gripper_tip, target)  -- closer = better
#   4. Update δq using a simple exponential moving average toward the
#      noise vector that gave the best reward so far.
#
# No neural network, no backprop, no external sensor -- just the FK model
# as the internal critic. Simple, fast, works on real hardware because the
# FK geometry matches the physical arm.

class PickPlaceLearner:
    def __init__(self, model, data, lock=None):
        self.m = model
        self.d = data
        self.lock = lock
        self.tip_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_PREFIX + j)
                for j in IK_JOINTS]
        self.qadr = [model.jnt_qposadr[j] for j in jids]
        self.dofadr = [model.jnt_dofadr[j] for j in jids]
        self.ranges = [model.jnt_range[j].copy() for j in jids]
        # Learned offsets, one 5-vector per square name (e.g. "e4")
        self.learned_offset = {}
        self.best_reward = {}

    # -- IK baseline ----------------------------------------------------------

    def ik(self, target_xyz, iters=300, damping=0.05, tol=1.5e-3):
        """Damped-least-squares IK (same approach as ArmController.solve_ik)."""
        saved = self.d.qpos.copy()
        q = np.array([self.d.qpos[a] for a in self.qadr])
        for _ in range(iters):
            for a, v in zip(self.qadr, q):
                self.d.qpos[a] = v
            mujoco.mj_forward(self.m, self.d)
            err = target_xyz - self.d.site_xpos[self.tip_sid]
            if np.linalg.norm(err) < tol:
                break
            jacp = np.zeros((3, self.m.nv))
            jacr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.tip_sid)
            J = jacp[:, self.dofadr]
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
            q = np.clip(q + dq,
                        [r[0] for r in self.ranges],
                        [r[1] for r in self.ranges])
        self.d.qpos[:] = saved
        mujoco.mj_forward(self.m, self.d)
        return q

    # -- FK eval (the "no external feedback" reward) --------------------------

    def fk_distance(self, q, target_xyz):
        """Apply q to a scratch copy of qpos, run FK, measure tip error."""
        saved = self.d.qpos.copy()
        for a, v in zip(self.qadr, q):
            self.d.qpos[a] = v
        mujoco.mj_forward(self.m, self.d)
        dist = float(np.linalg.norm(self.d.site_xpos[self.tip_sid] - target_xyz))
        self.d.qpos[:] = saved
        mujoco.mj_forward(self.m, self.d)
        return dist

    # -- One learning episode -------------------------------------------------

    def train_episode(self, square_name, file_idx, rank_idx,
                      noise_scale=0.04, lr=0.3):
        """
        One learning trial for a target square.

        Reward = -distance(gripper, target).  Positive = closer.
        The sign convention is standard RL: higher reward = better.

        Returns (q_used, reward, dist_cm).
        """
        target = square_xyz(file_idx, rank_idx)
        target[2] = BOARD_SURFACE_Z + GRASP_Z_OFFSET   # grasp height

        q_ik = self.ik(target)
        offset = self.learned_offset.get(square_name, np.zeros(5))

        noise = np.random.randn(5) * noise_scale
        q_try = np.clip(q_ik + offset + noise,
                        [r[0] for r in self.ranges],
                        [r[1] for r in self.ranges])

        dist = self.fk_distance(q_try, target)
        reward = -dist   # closer → higher reward (less negative)

        # Update learned offset if this trial beat the best known
        prev_best = self.best_reward.get(square_name, -1e9)
        if reward > prev_best:
            self.best_reward[square_name] = reward
            # Shift offset toward the noise that improved things
            self.learned_offset[square_name] = (1 - lr) * offset + lr * (offset + noise)

        return q_try, reward, dist * 100   # dist in cm

    # -- Full training run ----------------------------------------------------

    def train(self, square_name, file_idx, rank_idx, n_episodes=60):
        """Train pick-and-place for a single square. Prints a progress log."""
        print(f"\n  Training on square {square_name}  "
              f"({n_episodes} episodes, no external feedback)\n")
        print(f"  {'ep':>4s}  {'reward':>8s}  {'dist_cm':>8s}  {'note'}")

        for ep in range(n_episodes):
            q, rw, dist_cm = self.train_episode(square_name, file_idx, rank_idx)
            note = "← new best" if abs(rw - self.best_reward[square_name]) < 1e-9 else ""
            if ep % 10 == 0 or note:
                print(f"  {ep:>4d}  {rw:>8.4f}  {dist_cm:>8.3f} cm  {note}")

        best_dist = -self.best_reward[square_name] * 100
        print(f"\n  Done. Best error for {square_name}: {best_dist:.2f} cm")
        return self.learned_offset.get(square_name, np.zeros(5))


# ---------------------------------------------------------------------------
# REAL-HARDWARE OUTPUT (optional)
# ---------------------------------------------------------------------------

def send_to_hardware(q_joints, serial_port="/dev/ttyACM0", baud=115200):
    """
    Convert the 5 IK+learned joint angles (radians) to the 6-value servo
    command the Arduino expects (degrees, order: base,shoulder,arm,wrist,
    elbow,gripper) and send over serial.

    ⚠ The URDF's zero-angle convention is NOT the same as the servo's 90°
    neutral. You need per-joint calibration offsets before this gives correct
    physical motion -- see 'Also note' in README.md. This function shows the
    structure; fill in OFFSETS for your specific arm.
    """
    import serial  # pip install pyserial (already in Python_Control_Learm)

    # Placeholder calibration offsets (degrees to add to each raw radian->deg value)
    OFFSETS_DEG = [0, 0, 0, 0, 0]  # ← measure these by matching sim zero pose to real arm

    shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll = q_joints
    servo_deg = [
        np.degrees(shoulder_pan) + 90 + OFFSETS_DEG[0],   # base: 90 = centre
        np.degrees(shoulder_lift) + 90 + OFFSETS_DEG[1],  # shoulder
        np.degrees(elbow) + 90 + OFFSETS_DEG[2],           # arm
        np.degrees(wrist_flex) + 90 + OFFSETS_DEG[3],      # wrist
        np.degrees(wrist_roll) + 90 + OFFSETS_DEG[4],      # elbow/roll
        90,                                                  # gripper: 90 = open
    ]
    servo_deg = [int(np.clip(v, 0, 180)) for v in servo_deg]
    cmd = ",".join(str(v) for v in servo_deg) + "\n"
    print(f"  → hardware: {cmd.strip()}")
    with serial.Serial(serial_port, baud, timeout=1) as ser:
        time.sleep(0.05)
        ser.write(cmd.encode())


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pick-and-place geometry + online learner")
    parser.add_argument("--distances", action="store_true",
                        help="print every square's distance from the arm base")
    parser.add_argument("--square", default="e4",
                        help="train pick-and-place on this square (default: e4)")
    parser.add_argument("--episodes", type=int, default=60,
                        help="number of learning episodes (default: 60)")
    parser.add_argument("--serial", action="store_true",
                        help="also send best config to real hardware over serial")
    args = parser.parse_args()

    if args.distances:
        print_board_distances()
        raise SystemExit

    # Parse square name (e.g. "e4" -> file_idx=4, rank_idx=3)
    sq = args.square.lower()
    if len(sq) != 2 or sq[0] not in FILES or not sq[1].isdigit():
        raise ValueError(f"invalid square: {sq!r}  -- use a letter a-h + digit 1-8, e.g. e4")
    file_idx = FILES.index(sq[0])
    rank_idx = int(sq[1]) - 1

    # Show geometry first
    target_surface = square_xyz(file_idx, rank_idx)
    d = distance_breakdown(target_surface)
    print(f"\nSquare {sq.upper()} geometry:")
    print(f"  World xyz        : {target_surface}")
    print(f"  Horizontal reach : {d['horizontal_cm']} cm", end="")
    print(f"  (within reach: {'YES' if d['within_reach'] else 'NO !'})")
    print(f"  Arm needs to go  : Δx={d['dx']*100:.1f} cm  Δy={d['dy']*100:.1f} cm  Δz={d['dz']*100:.1f} cm")

    # Load sim model
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d_obj = mujoco.MjData(m)
    mujoco.mj_forward(m, d_obj)

    learner = PickPlaceLearner(m, d_obj)
    best_offset = learner.train(sq, file_idx, rank_idx, n_episodes=args.episodes)
    print(f"\n  Learned offset (radians): {np.round(best_offset, 4)}")

    if args.serial:
        q_ik = learner.ik(target_surface)
        q_final = q_ik + best_offset
        print("\nSending best config to hardware:")
        send_to_hardware(q_final)
