void setup() {
  pinMode(10, OUTPUT);
}
void loop() {
  digitalWrite(10, HIGH);
  delay(2000); // 維持高電位 2 秒
  digitalWrite(10, LOW);
  delay(2000); // 維持低電位 2 秒
}