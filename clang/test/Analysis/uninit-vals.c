// RUN: clang -warn-uninit-values -verify %s

int f1() {
  int x;
  return x; // expected-warning{use of uninitialized variable}
}

int f2(int x) {
  int y;
  int z = x + y; // expected-warning {use of uninitialized variable}
  return z;
}


int f3(int x) {
  int y;
  return x ? 1 : y; // expected-warning {use of uninitialized variable}
}

int f4(int x) {
  int y;
  if (x) y = 1;
  return y; // no-warning
}

int f5() {
  int a;
  a = 30; // no-warning
}
