

union foo { int X; };

int test(union foo* F) {
  {
    union foo { float X; } A;
  }
}
