// RUN: false
// XFAIL: *
int f1() {
  int y = 1;
  y++;
  return y;
}

void f2() {
  int x = 1;
  x = f1();
  if (x == 1) {
    int *p = 0;
    *p = 3; // no-warning
  }
  if (x == 2) {
    int *p = 0;
    *p = 3; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}}
  }
}
