#include <Servo.h>

Servo baseServo;
Servo shoulderServo;
Servo armServo;
Servo wristServo;
Servo extraServo;   // = elbow
Servo gripperServo;

// PIN MAPPING (physical wiring)
// pin 3=gripper, 5=wrist, 6=arm, 9=elbow/extra, 10=shoulder, 11=base
const int BASE_PIN     = 11;
const int SHOULDER_PIN = 10;
const int ARM_PIN      = 6;
const int WRIST_PIN    = 5;
const int EXTRA_PIN    = 9;   // elbow
const int GRIPPER_PIN  = 3;

// Current positions  (order: base, shoulder, arm, wrist, extra, gripper)
int pos[6] = {90, 90, 90, 90, 90, 90};
// Target positions
int target[6] = {90, 90, 90, 90, 90, 90};
// Limits
int minLim[6] = {0, 30, 30, 0, 0, 20};
int maxLim[6] = {180, 150, 150, 180, 180, 160};

String input = "";

void setup() {
  Serial.begin(115200);
  baseServo.attach(BASE_PIN);
  shoulderServo.attach(SHOULDER_PIN);
  armServo.attach(ARM_PIN);
  wristServo.attach(WRIST_PIN);
  extraServo.attach(EXTRA_PIN);
  gripperServo.attach(GRIPPER_PIN);
  writeAllServos();
  delay(1000);
  Serial.println("6DOF robotic arm ready.");
}

void loop() {
  readSerialCommand();
  smoothMoveServos();
  writeAllServos();
  delay(15);
}

void readSerialCommand() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      parseCommand(input);
      input = "";
    }
    else {
      input += c;
    }
  }
}

// Format:
// 90,90,90,90,90,90
void parseCommand(String cmd) {
  int index = 0;
  int lastComma = -1;
  for (int i = 0; i <= cmd.length(); i++) {
    if (i == cmd.length() || cmd.charAt(i) == ',') {
      if (index < 6) {
        String piece = cmd.substring(lastComma + 1, i);
        target[index] = piece.toInt();
        target[index] =
          constrain(target[index], minLim[index], maxLim[index]);
        index++;
      }
      lastComma = i;
    }
  }
}

void smoothMoveServos() {
  for (int i = 0; i < 6; i++) {
    if (pos[i] < target[i]) {
      pos[i]++;
    }
    else if (pos[i] > target[i]) {
      pos[i]--;
    }
  }
}

void writeAllServos() {
  baseServo.write(pos[0]);
  shoulderServo.write(pos[1]);
  armServo.write(pos[2]);
  wristServo.write(pos[3]);
  extraServo.write(pos[4]);
  gripperServo.write(pos[5]);
}
