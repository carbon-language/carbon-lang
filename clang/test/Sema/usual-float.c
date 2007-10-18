// RUN: clang %s -fsyntax-only

void foo(int dir, int n, int tindex) {
  const float PI = 3.142;
float ang = (float) tindex * (-dir*2.0f*PI/n);
}
