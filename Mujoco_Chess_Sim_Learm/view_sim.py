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
   actually is and rejected if it doesn't match. After your move, Stockfish
   (playing "the robot's" side) replies automatically. Every move is
   animated by picking the corresponding chess-piece body up off the board
   and setting it down on its destination square -- this is a kinematic
   replay of the game state, not an IK-planned pick-and-place by the arm's
   own joints (the arm has no collision geometry yet and isn't driven
   during play mode). Teaching the arm to actually execute these moves
   (inverse kinematics, grasping, maybe RL) is flagged as follow-up work,
   not attempted here.

   Requires the `chess` package (see requirements.txt) and a `stockfish`
   binary on PATH (macOS: `brew install stockfish`), or pass
   --stockfish /path/to/stockfish.
"""

import argparse
import os
import shutil
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "learm_chess_scene.xml"
ARM_PREFIX = "arm_"
PIECE_PREFIX = "piece_"
TABLE_SURFACE_Z = 0.0  # see table_scene.xml: "Table: top surface defined at world z = 0"

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

    def apply_move(self, board, move, mover_label):
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
                self._animate(body, xy, TABLE_SURFACE_Z)

        src, dst = chess.square_name(move.from_square), chess.square_name(move.to_square)
        body = self.square_to_body.pop(src)
        target = self.square_xyz(dst)
        self._animate(body, target[:2], target[2])
        self.square_to_body[dst] = body

        if board.is_castling(move):
            rook_pairs = {"g1": ("h1", "f1"), "c1": ("a1", "d1"), "g8": ("h8", "f8"), "c8": ("a8", "d8")}
            rook_src, rook_dst = rook_pairs[dst]
            rbody = self.square_to_body.pop(rook_src)
            rtarget = self.square_xyz(rook_dst)
            self._animate(rbody, rtarget[:2], rtarget[2])
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

    viewer = mujoco.viewer.launch_passive(model, data)
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
        table.apply_move(board, result.move, "Robot (Stockfish)")

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
