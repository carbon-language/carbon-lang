#include <stdio.h>

struct sometimes {
  short offset; short bit;
  short live_length; short calls_crossed;
} Y;

int main() {
  int X;
  {
    struct sometimes { int X, Y; } S;
    S.X = 1;
    X = S.X;
  }
  { 
    struct sometimes { char X; } S;
    S.X = -1;
    X += S.X;
  }
  X += Y.offset;

  printf("Result is %d\n", X);
  return X;
}
