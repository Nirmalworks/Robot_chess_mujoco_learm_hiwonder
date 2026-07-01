import time
import serial
import pygame

# =========================
# SERIAL SETTINGS
# =========================
COM_PORT = "/dev/ttyACM0"
BAUD = 115200

ser = serial.Serial(COM_PORT, BAUD, timeout=1)
time.sleep(2)

# =========================
# SERVO ORDER
# base, shoulder, arm, wrist, extra(elbow), gripper
# =========================

HOME = [90, 90, 90, 90, 90, 90]

GRIPPER_OPEN = 80
GRIPPER_CLOSED = 143

# Limits (match Arduino constrain values)
MIN_LIM = [0, 30, 30, 0, 0, 20]
MAX_LIM = [180, 150, 150, 180, 180, 160]

# Live position we update with the controller
pos = HOME.copy()

STEP = 2          # degrees per loop when control held
DEADZONE = 0.25   # ignore tiny stick drift

# =========================
# FUNCTIONS
# =========================

def send_pos(p):
    command = ",".join(str(int(v)) for v in p) + "\n"
    ser.write(command.encode())

def clamp_all():
    for i in range(6):
        if pos[i] < MIN_LIM[i]:
            pos[i] = MIN_LIM[i]
        if pos[i] > MAX_LIM[i]:
            pos[i] = MAX_LIM[i]

def dz(val):
    return 0 if abs(val) < DEADZONE else val

# =========================
# INIT CONTROLLER
# =========================

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller found. Plug in your Xbox controller.")
    ser.close()
    raise SystemExit

js = pygame.joystick.Joystick(0)
print("Controller connected:", js.get_name())

# Start at home
send_pos(pos)
time.sleep(1)

print("Each servo on its own control:")
print("  L stick X = base      L stick Y = shoulder")
print("  R stick Y = arm       R stick X = wrist")
print("  LT / RT   = elbow     X = open / B = close gripper")
print("  Y = home              Back = quit")
print("  Start     = PRINT POSITION (copy this to notepad)")

# =========================
# CONTROL LOOP
# =========================

clock = pygame.time.Clock()
running = True

try:
    while running:
        pygame.event.pump()

        # --- STICKS (one joint each) ---
        lx = dz(js.get_axis(0))   # left stick X  -> base
        ly = dz(js.get_axis(1))   # left stick Y  -> shoulder
        rx = dz(js.get_axis(3))   # right stick X -> wrist
        ry = dz(js.get_axis(4))   # right stick Y -> arm

        pos[0] += STEP * lx       # base
        pos[1] += STEP * ly       # shoulder
        pos[3] += STEP * rx       # wrist
        pos[2] += STEP * ry       # arm

        # --- TRIGGERS -> elbow (extra, index 4) ---
        # Xbox triggers: -1.0 released, +1.0 pressed. Treat > 0.2 as pressed.
        lt = js.get_axis(2)
        rt = js.get_axis(5)
        if rt > 0.2:
            pos[4] += STEP        # elbow one way
        if lt > 0.2:
            pos[4] -= STEP        # elbow other way

        # --- BUTTONS -> gripper (index 5) ---
        if js.get_button(3):      # X = open gripper
            pos[5] = GRIPPER_OPEN
        if js.get_button(1):      # B = close gripper
            pos[5] = GRIPPER_CLOSED

        # --- HOME / QUIT / PRINT ---
        if js.get_button(4):      # Y = go home
            pos[:] = HOME.copy()
        if js.get_button(7):      # Back = quit
            running = False
        if js.get_button(8):      # Start = print position for notepad
            print(f"\n>>> POSITION: base={int(pos[0])}, shoulder={int(pos[1])}, arm={int(pos[2])}, wrist={int(pos[3])}, elbow={int(pos[4])}, gripper={int(pos[5])}")
            print(f">>> RAW LIST: [{int(pos[0])}, {int(pos[1])}, {int(pos[2])}, {int(pos[3])}, {int(pos[4])}, {int(pos[5])}]\n")

        # --- DEBUG: print activity ---
        if lx or ly or rx or ry:
            print(f"Sticks -> base:{lx:+.2f} shoulder:{ly:+.2f} wrist:{rx:+.2f} arm:{ry:+.2f}")
        if rt > 0.2:
            print("RT -> elbow +")
        if lt > 0.2:
            print("LT -> elbow -")
        if js.get_button(3):
            print("X -> gripper open")
        if js.get_button(1):
            print("B -> gripper close")
        if js.get_button(4):
            print("Y -> home")

        clamp_all()
        send_pos(pos)

        clock.tick(50)            # 50 Hz loop

finally:
    pygame.quit()
    ser.close()
