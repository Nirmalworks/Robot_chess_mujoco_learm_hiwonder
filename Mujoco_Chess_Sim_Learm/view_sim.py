"""Launch the interactive MuJoCo viewer on the LeArm + chess table scene.

Two modes:

1. Manual jogging (default) -- run:

       python3 view_sim.py

   Open the right-hand "Control" tab to find one slider per joint
   (shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll, grip_left) --
   drag a slider to jog that joint. The gripper fingers mirror grip_left
   automatically. This only drives the simulation; see README.md for why it
   does NOT move the real hardware.

   Hold Ctrl and drag with the mouse to apply a force/torque to a body, or
   double-click a body to select it (its name + live xyz/quat show in the
   bottom-left overlay) -- this is the easiest way to read off a position
   when you jog the arm onto square a1 or h8 for the corner-calibration
   method in board_calibration.py.

2. Play chess against Stockfish -- run:

       python3 view_sim.py --play
       python3 view_sim.py --play --side black
       python3 view_sim.py --play --skill 5 --movetime 0.5

   Type moves in standard algebraic notation at the prompt (e.g. "e4",
   "Nf3", "exd5", "O-O"), optionally followed by a color word as a
   sanity-check ("e4 white") -- it's compared against whose turn it
   actually is and rejected if it doesn't match.

   Your move is just registered (matching how this would work on the real
   board: you physically moved your own piece by hand). Stockfish then
   replies as "the robot," and its move is carried out by the arm itself --
   numerical inverse kinematics (damped-least-squares on the gripper_tip
   site's Jacobian, see ArmController) drives shoulder_pan/shoulder_lift/
   elbow/wrist_flex/wrist_roll through a hover -> descend -> grasp -> lift
   -> carry -> place -> release -> retreat sequence, with the position
   actuators' built-in PD control giving the "slowly move" motion. There's
   no real grasp physics (the arm has no collision geometry yet -- see
   Known limitations in README.md), so the piece is kinematically pinned to
   the gripper_tip site while "held," not actually gripped by friction.

   Requires the `chess` package (see requirements.txt) and a `stockfish`
   binary on PATH (macOS: `brew install stockfish`), or pass
   --stockfish /path/to/stockfish.
"""

import argparse
import os
import shutil
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "learm_chess_scene.xml"
ARM_PREFIX = "arm_"
PIECE_PREFIX = "piece_"
TABLE_SURFACE_Z = 0.0  # see table_scene.xml: "Table: top surface defined at world z = 0"

IK_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll"]

# IK starting seed that points the arm TOWARD the board.
# The URDF zero pose is fully extended upward, and REST_POSE points the arm
# backward (-x). With REST_POSE as the IK seed the solver hits joint limits
# immediately and stalls ~190 mm from any board square.
# This seed (shoulder_lift>0, elbow<0) folds the arm forward and down;
# IK converges to <2 mm for all 64 squares in under 10 iterations.
IK_BOARD_SEED = [0.0, 0.65, -1.22, -0.73, 0.0]

HOVER_HEIGHT = 0.06   # transit clearance above a square's surface -- clears the tallest piece (king)
GRASP_HEIGHT = 0.015  # gripper_tip height above a square's surface while "holding" a piece
GRIP_OPEN = 0.0
GRIP_CLOSED = -1.2    # grip_left's URDF range is [-1.57, 0]; doesn't need to be physically exact
                       # since there's no real grasp contact -- see module docstring.

# A neutral "parked" pose for the arm during chess play, so it isn't sitting
# in the URDF's zero pose (fully extended straight up, see README) the whole
# game. Not derived from any IK target -- just a reasonable resting pose.
REST_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.6,
    "elbow": 1.0,
    "wrist_flex": 0.3,
    "wrist_roll": 0.0,
    "grip_left": 0.0,
}


def find_stockfish(explicit):
    if explicit:
        return explicit
    found = shutil.which("stockfish")
    if found:
        return found
    for guess in ("/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish"):
        if os.path.exists(guess):
            return guess
    return None


def site_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def body_joint_addrs(model, body_name):
    """Return (qpos_addr, dof_addr) of a free-jointed body's joint."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jadr = model.body_jntadr[bid]
    return model.jnt_qposadr[jadr], model.jnt_dofadr[jadr]


class ArmController:
    """Numerical IK + PD-actuator pick-and-place for the 5-DOF arm chain.

    Position-only damped-least-squares IK on the gripper_tip site's
    Jacobian (no orientation constraint -- the arm has only 5 DOF feeding a
    3-DOF position task, so it's already redundant; adding an orientation
    task would need per-axis sign analysis of the URDF that isn't worth the
    risk for a cosmetic pick-and-place). solve_ik() runs its iterations
    against the live `data` under `lock`, then restores the original qpos
    -- it never disturbs the rendered/stepped state, it only computes a
    target for the position actuators to converge to over real time.
    """

    def __init__(self, model, data, lock):
        self.model = model
        self.data = data
        self.lock = lock
        self.tip_sid = site_id(model, "gripper_tip")
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_PREFIX + j) for j in IK_JOINTS]
        self.qadr = [model.jnt_qposadr[j] for j in jids]
        self.dofadr = [model.jnt_dofadr[j] for j in jids]
        self.ranges = [model.jnt_range[j].copy() for j in jids]
        self.grip_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_grip_left")
        self.arm_actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_" + j) for j in IK_JOINTS]

    def solve_ik(self, target, q0=None, iters=300, damping=0.05, step=1.0, tol=1.5e-3):
        target = np.asarray(target, dtype=float)
        with self.lock:
            q = np.array([self.data.qpos[a] for a in self.qadr]) if q0 is None else np.array(q0, dtype=float)
            saved_qpos = self.data.qpos.copy()
            err_norm = None
            for _ in range(iters):
                for a, v in zip(self.qadr, q):
                    self.data.qpos[a] = v
                mujoco.mj_forward(self.model, self.data)
                err = target - self.data.site_xpos[self.tip_sid]
                err_norm = float(np.linalg.norm(err))
                if err_norm < tol:
                    break
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tip_sid)
                J = jacp[:, self.dofadr]
                dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
                q = q + step * dq
                for k, (lo, hi) in enumerate(self.ranges):
                    q[k] = np.clip(q[k], lo, hi)
            self.data.qpos[:] = saved_qpos
            mujoco.mj_forward(self.model, self.data)
        return q, err_norm

    def set_gripper(self, value, max_step=0.04, poll_dt=0.02, timeout=2.0):
        """Ramp grip_left's ctrl toward `value` in small steps. A sudden
        jump (even with the arm otherwise still) drives the mimic-joint
        equality constraints (grip_right/tendon/finger, see build_scene.py)
        into a one-step violation large enough to blow up qacc -- verified
        this empirically, same root cause as the arm-joint ramping in
        _ramp_ctrl_to.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                cur = self.data.ctrl[self.grip_actuator]
                step = np.clip(value - cur, -max_step, max_step)
                new = cur + step
                self.data.ctrl[self.grip_actuator] = new
            if abs(new - value) < 1e-9:
                break
            time.sleep(poll_dt)

    def _ramp_ctrl_to(self, q_target, follow_body=None, follow_offset=(0, 0, 0),
                       max_step=0.035, tol=0.03, timeout=6.0, poll_dt=0.02):
        """Ramp `ctrl` toward q_target in small steps (rather than jumping
        straight to it) so the PD position actuators -- high gain, very
        light/approximate link masses, see README's "Approximate inertials"
        limitation -- don't see a huge instantaneous error and spike into
        instability. This also happens to be exactly how the real Arduino
        sketch eases servos (1deg/15ms, see root README), so it doubles as a
        more realistic "slowly move."
        """
        fbody = body_joint_addrs(self.model, follow_body) if follow_body else None
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                cur_ctrl = np.array([self.data.ctrl[aid] for aid in self.arm_actuators])
                step = np.clip(q_target - cur_ctrl, -max_step, max_step)
                new_ctrl = cur_ctrl + step
                for aid, v in zip(self.arm_actuators, new_ctrl):
                    self.data.ctrl[aid] = v
                cur_qpos = np.array([self.data.qpos[a] for a in self.qadr])
                if fbody:
                    tip = self.data.site_xpos[self.tip_sid].copy()
                    fqadr, fdadr = fbody
                    self.data.qpos[fqadr:fqadr + 3] = tip + np.asarray(follow_offset)
                    self.data.qpos[fqadr + 3:fqadr + 7] = [1, 0, 0, 0]
                    self.data.qvel[fdadr:fdadr + 6] = 0
            ctrl_settled = np.max(np.abs(new_ctrl - q_target)) < 1e-9
            if ctrl_settled and np.max(np.abs(cur_qpos - q_target)) < tol:
                break
            time.sleep(poll_dt)

    def move_to(self, target_xyz, follow_body=None, follow_offset=(0, 0, 0)):
        # Always seed IK from IK_BOARD_SEED (points arm toward board).
        # Using the current qpos as seed fails when coming from REST_POSE
        # because REST_POSE aims the arm backward — the solver stalls at joint
        # limits ~190 mm from any board square.
        q_target, err = self.solve_ik(target_xyz, q0=IK_BOARD_SEED)
        self._ramp_ctrl_to(q_target, follow_body=follow_body, follow_offset=follow_offset)
        return q_target, err

    def _move_vertical(self, start_xyz, end_xyz, n_steps=8,
                        follow_body=None, follow_offset=(0, 0, 0)):
        """Move strictly vertically (XY locked, Z interpolated) through n_steps
        IK waypoints.

        Why: move_to() targets a single Cartesian point but resolves it via IK
        into joint-space, so the path between waypoints curves in Cartesian
        space. That's fine for horizontal transit above the board (nothing to
        hit up there) but NOT for descent/ascent near the board surface, where
        a small horizontal swing would clip adjacent pieces. Stepping through
        many intermediate points with the same XY keeps the Cartesian path
        essentially straight -- each step's IK error is tiny so the joint
        correction is tiny, leaving no room for lateral drift.
        """
        start_xyz = np.asarray(start_xyz, dtype=float)
        end_xyz   = np.asarray(end_xyz,   dtype=float)
        for i in range(1, n_steps + 1):
            t = i / n_steps
            # X and Y come from start (locked), only Z interpolates
            waypoint = np.array([start_xyz[0], start_xyz[1],
                                  start_xyz[2] + (end_xyz[2] - start_xyz[2]) * t])
            self.move_to(waypoint, follow_body=follow_body,
                         follow_offset=follow_offset)

    def pick_and_place(self, body_name, src_xyz, dst_xyz):
        """
        Safe pick-and-place sequence -- gripper state at every phase:

          Phase 1 OPEN  → transit to hover above source (XY move, high up)
          Phase 2 OPEN  → descend VERTICALLY onto source (XY locked!)
          Phase 3 CLOSE → gripper closes around piece
          Phase 4 CLOSE → ascend VERTICALLY to hover height (XY locked!)
          Phase 5 CLOSE → transit to hover above destination (XY move, high up)
          Phase 6 CLOSE → descend VERTICALLY onto destination (XY locked!)
          Phase 7 OPEN  → release piece at destination
          Phase 8 OPEN  → ascend VERTICALLY back to hover (XY locked!)

        Gripper only changes state at grasp height -- never mid-transit and
        never at hover height where nearby pieces are directly below.
        Both descents and ascents use _move_vertical so the arm path stays
        straight up/down, not curving through neighboring squares.
        """
        src_xyz = np.asarray(src_xyz, dtype=float)
        dst_xyz = np.asarray(dst_xyz, dtype=float)
        hover_src = src_xyz + [0, 0, HOVER_HEIGHT]
        grasp_src = src_xyz + [0, 0, GRASP_HEIGHT]
        hover_dst = dst_xyz + [0, 0, HOVER_HEIGHT]
        grasp_dst = dst_xyz + [0, 0, GRASP_HEIGHT]
        follow_offset = (0, 0, -GRASP_HEIGHT)

        # Phase 1 -- gripper OPEN, transit at safe height
        print(f"  [1/8] OPEN  → hover above {body_name}...", flush=True)
        self.set_gripper(GRIP_OPEN)
        self.move_to(hover_src)

        # Phase 2 -- gripper OPEN, straight-down descent (XY locked)
        print(f"  [2/8] OPEN  → descend vertically onto {body_name}...", flush=True)
        self._move_vertical(hover_src, grasp_src)

        # Phase 3 -- close gripper AT grasp height
        print(f"  [3/8] CLOSING gripper around {body_name}...", flush=True)
        self.set_gripper(GRIP_CLOSED)
        time.sleep(0.25)

        # Phase 4 -- gripper CLOSED, straight-up ascent (XY locked, piece rides along)
        print(f"  [4/8] CLOSED → ascend vertically (piece lifted)...", flush=True)
        self._move_vertical(grasp_src, hover_src,
                            follow_body=body_name, follow_offset=follow_offset)

        # Phase 5 -- gripper CLOSED, transit at safe height
        print(f"  [5/8] CLOSED → carry to hover above destination...", flush=True)
        self.move_to(hover_dst, follow_body=body_name, follow_offset=follow_offset)

        # Phase 6 -- gripper CLOSED, straight-down descent to destination (XY locked)
        print(f"  [6/8] CLOSED → descend vertically to destination...", flush=True)
        self._move_vertical(hover_dst, grasp_dst,
                            follow_body=body_name, follow_offset=follow_offset)

        # Snap piece qpos to exact destination (no real contact physics yet)
        qadr, dadr = body_joint_addrs(self.model, body_name)
        with self.lock:
            self.data.qpos[qadr:qadr + 3] = dst_xyz
            self.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
            self.data.qvel[dadr:dadr + 6] = 0

        # Phase 7 -- open gripper AT destination (piece released)
        print(f"  [7/8] OPENING gripper, releasing piece...", flush=True)
        self.set_gripper(GRIP_OPEN)
        time.sleep(0.2)

        # Phase 8 -- gripper OPEN, straight-up retreat (XY locked, no accidental sweeps)
        print(f"  [8/8] OPEN   → ascend vertically, retreating...", flush=True)
        self._move_vertical(grasp_dst, hover_dst)

    def go_rest(self):
        print("  arm: returning to rest pose...", flush=True)
        q = np.array([REST_POSE[j] for j in IK_JOINTS])
        self._ramp_ctrl_to(q, timeout=6.0)
        self.set_gripper(REST_POSE["grip_left"])


class ChessTable:
    """Bridges a python-chess Board to the piece bodies in the MuJoCo scene."""

    def __init__(self, model, data, lock):
        self.model = model
        self.data = data
        self.lock = lock
        self.square_to_body = {}
        for i in range(model.nbody):
            name = model.body(i).name
            if name.startswith(PIECE_PREFIX):
                self.square_to_body[name[len(PIECE_PREFIX):]] = name
        self.captured_count = {"white": 0, "black": 0}
        self._board_bounds = self._compute_board_bounds()

    def _compute_board_bounds(self):
        with self.lock:
            a1 = self.data.site_xpos[site_id(self.model, "sq_a1")].copy()
            h8 = self.data.site_xpos[site_id(self.model, "sq_h8")].copy()
        spacing = abs(h8[0] - a1[0]) / 7.0
        min_x, max_x = sorted([a1[0], h8[0]])
        min_y, max_y = sorted([a1[1], h8[1]])
        return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "spacing": spacing}

    def square_xyz(self, square_name):
        with self.lock:
            return self.data.site_xpos[site_id(self.model, f"sq_{square_name}")].copy()

    def _graveyard_xy(self, color):
        k = self.captured_count[color]
        self.captured_count[color] += 1
        b = self._board_bounds
        col, row = k % 8, k // 8
        x = b["min_x"] + b["spacing"] * col
        if color == "white":
            y = b["max_y"] + b["spacing"] * (1.0 + row)
        else:
            y = b["min_y"] - b["spacing"] * (1.0 + row)
        return x, y

    def _animate(self, body_name, target_xy, target_z, lift=0.035, steps=30, dt=0.012):
        qadr, dadr = body_joint_addrs(self.model, body_name)
        with self.lock:
            start = self.data.qpos[qadr:qadr + 3].copy()
        target_xy = np.asarray(target_xy, dtype=float)
        for i in range(1, steps + 1):
            t = i / steps
            xy = start[:2] + (target_xy - start[:2]) * t
            z = start[2] + (target_z - start[2]) * t + lift * np.sin(np.pi * t)
            with self.lock:
                self.data.qpos[qadr:qadr + 2] = xy
                self.data.qpos[qadr + 2] = z
                self.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
                self.data.qvel[dadr:dadr + 6] = 0
            time.sleep(dt)

    def _move_body(self, arm, body, dst_xyz):
        """Move `body` (currently at its own qpos) to dst_xyz -- via arm IK
        pick-and-place if `arm` is given, otherwise the simple kinematic
        lift-carry-place animation used for the human's own moves."""
        if arm is not None:
            qadr, _ = body_joint_addrs(self.model, body)
            with self.lock:
                src_xyz = self.data.qpos[qadr:qadr + 3].copy()
            arm.pick_and_place(body, src_xyz, dst_xyz)
        else:
            self._animate(body, dst_xyz[:2], dst_xyz[2])

    def apply_move(self, board, move, mover_label, arm=None):
        import chess

        capture_sq = None
        if board.is_en_passant(move):
            capture_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        elif board.is_capture(move):
            capture_sq = move.to_square

        if capture_sq is not None:
            captured_piece = board.piece_at(capture_sq)
            color = "white" if captured_piece.color == chess.WHITE else "black"
            csq = chess.square_name(capture_sq)
            body = self.square_to_body.pop(csq, None)
            if body:
                xy = self._graveyard_xy(color)
                # CAPTURE STEP 1 -- clear the captured piece to the graveyard
                # BEFORE the attacking piece moves. Otherwise the arm would try
                # to place two pieces on the same square simultaneously.
                print(f"\n  *** CAPTURE: picking up {color} piece on {csq} → graveyard ***",
                      flush=True)
                self._move_body(arm, body, np.array([xy[0], xy[1], TABLE_SURFACE_Z]))
                print(f"  *** Captured piece cleared. Now moving attacker. ***\n", flush=True)

        src, dst = chess.square_name(move.from_square), chess.square_name(move.to_square)
        body = self.square_to_body.pop(src)
        target = self.square_xyz(dst)
        self._move_body(arm, body, target)
        self.square_to_body[dst] = body

        if board.is_castling(move):
            rook_pairs = {"g1": ("h1", "f1"), "c1": ("a1", "d1"), "g8": ("h8", "f8"), "c8": ("a8", "d8")}
            rook_src, rook_dst = rook_pairs[dst]
            rbody = self.square_to_body.pop(rook_src)
            rtarget = self.square_xyz(rook_dst)
            self._move_body(arm, rbody, rtarget)
            self.square_to_body[rook_dst] = rbody

        if move.promotion:
            print(f"  (note: {dst} is now a {chess.piece_name(move.promotion)} in the engine's "
                  f"bookkeeping/Stockfish's eyes -- the 3-D piece keeps its pawn shape, a known "
                  f"cosmetic limitation that doesn't affect game logic.)")

        san = board.san(move)
        board.push(move)
        print(f"{mover_label}: {san}")


def play_chess(model, data, args):
    import chess
    import chess.engine

    sf_path = find_stockfish(args.stockfish)
    if not sf_path:
        print("Stockfish binary not found. Install it (e.g. `brew install stockfish` on macOS / "
              "`apt install stockfish` on Linux) or pass --stockfish /path/to/stockfish.")
        return

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    if args.skill is not None:
        try:
            engine.configure({"Skill Level": args.skill})
        except chess.engine.EngineError:
            print(f"(this Stockfish build doesn't support 'Skill Level' -- ignoring --skill {args.skill})")

    board = chess.Board()
    human_color = chess.WHITE if args.side == "white" else chess.BLACK

    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        if "mjpython" in str(e):
            print("On macOS, --play needs the special `mjpython` launcher instead of `python3`\n"
                  "(MuJoCo's passive viewer requires the main thread on macOS). Run:\n\n"
                  f"    mjpython {' '.join([__file__] + sys.argv[1:])}\n")
        else:
            print(f"Couldn't open the MuJoCo viewer: {e}")
        engine.quit()
        return

    lock = threading.Lock()
    stop = threading.Event()

    with lock:
        for jname, val in REST_POSE.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_PREFIX + jname)
            data.qpos[model.jnt_qposadr[jid]] = val
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_" + jname)
            data.ctrl[aid] = val
        mujoco.mj_forward(model, data)

    table = ChessTable(model, data, lock)
    arm = ArmController(model, data, lock)

    def physics_loop():
        while viewer.is_running() and not stop.is_set():
            t0 = time.time()
            with lock:
                mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)

    thread = threading.Thread(target=physics_loop, daemon=True)
    thread.start()

    def play_engine_move():
        result = engine.play(board, chess.engine.Limit(time=args.movetime))
        table.apply_move(board, result.move, "Robot (Stockfish)", arm=arm)
        arm.go_rest()

    print(f"\nYou are playing {args.side}. Stockfish plays the other side (\"the robot\").")
    print("Enter moves in SAN, e.g. 'e4', 'Nf3', 'exd5', 'O-O' -- optionally followed by a color\n"
          "word as a sanity check, e.g. 'e4 white'. Type 'quit' to stop.\n")

    try:
        if human_color == chess.BLACK:
            print("Stockfish (White) moves first...")
            play_engine_move()
        print(board.unicode(borders=True))

        while not board.is_game_over() and viewer.is_running():
            turn_color = "White" if board.turn == chess.WHITE else "Black"
            try:
                raw = input(f"[{turn_color} to move] > ").strip()
            except EOFError:
                break
            if not raw:
                continue
            if raw.lower() in ("quit", "exit"):
                break

            parts = raw.split()
            move_text = parts[0]
            if len(parts) > 1 and parts[1].lower() in ("white", "black"):
                expected = chess.WHITE if parts[1].lower() == "white" else chess.BLACK
                if expected != board.turn:
                    print(f"  It's actually {turn_color}'s turn, not {parts[1]}. Move ignored.")
                    continue

            move = None
            try:
                move = board.parse_san(move_text)
            except ValueError:
                try:
                    candidate = chess.Move.from_uci(move_text)
                    if candidate in board.legal_moves:
                        move = candidate
                except ValueError:
                    pass
            if move is None:
                print(f"  '{move_text}' isn't a legal move from here. Try SAN like 'e4', 'Nf3', 'O-O'.")
                continue

            table.apply_move(board, move, "You")
            if board.is_game_over():
                break

            play_engine_move()
            print(board.unicode(borders=True))

        if board.is_game_over():
            print(f"\nGame over: {board.result()} ({board.outcome().termination.name})")
    finally:
        stop.set()
        thread.join(timeout=1.0)
        engine.quit()
        if viewer.is_running():
            viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--play", action="store_true", help="interactive chess vs Stockfish; pieces move in the MuJoCo scene")
    parser.add_argument("--side", choices=["white", "black"], default="white", help="which color you play (default: white)")
    parser.add_argument("--stockfish", default=None, help="path to the stockfish binary (auto-detected on PATH/Homebrew if omitted)")
    parser.add_argument("--movetime", type=float, default=1.0, help="seconds Stockfish thinks per move (default: 1.0)")
    parser.add_argument("--skill", type=int, default=None, help="Stockfish 'Skill Level' 0-20 (default: full strength)")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    if args.play:
        play_chess(model, data, args)
    else:
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
