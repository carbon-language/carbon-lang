struct sometimes {
  short offset; short bit;
  short live_length; short calls_crossed;
} Y;

int main() {
  struct sometimes { int X, Y; } S;
  S.X = 1;
  return Y.offset;
}
