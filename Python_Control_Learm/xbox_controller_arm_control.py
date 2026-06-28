import time
import serial
import pygame

# =========================
# SERIAL SETTINGS
# =========================
COM_PORT = "COM4"
BAUD = 115200

ser = serial.Serial(COM_PORT, BAUD, timeout=1)
time.sleep(2)

# =========================
# SERVO ORDER
# base, shoulder, arm, wrist, extra(elbow), gripper
# =========================

HOME = [100, 85, 84, 178, 72, 143]

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
        if js.get_button(2):      # X = open gripper
            pos[5] = GRIPPER_OPEN
        if js.get_button(1):      # B = close gripper
            pos[5] = GRIPPER_CLOSED

        # --- HOME / QUIT ---
        if js.get_button(3):      # Y = go home
            pos[:] = HOME.copy()
        if js.get_button(6):      # Back = quit
            running = False

        # --- DEBUG: print activity ---
        if lx or ly or rx or ry:
            print(f"Sticks -> base:{lx:+.2f} shoulder:{ly:+.2f} wrist:{rx:+.2f} arm:{ry:+.2f}")
        if rt > 0.2:
            print("RT -> elbow +")
        if lt > 0.2:
            print("LT -> elbow -")
        if js.get_button(2):
            print("X -> gripper open")
        if js.get_button(1):
            print("B -> gripper close")
        if js.get_button(3):
            print("Y -> home")

        clamp_all()
        send_pos(pos)

        clock.tick(50)            # 50 Hz loop

finally:
    pygame.quit()
    ser.close()
