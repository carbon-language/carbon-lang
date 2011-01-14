// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -analyzer-inline-call -analyzer-store region -verify %s

int test1_f1() {
  int y = 1;
  y++;
  return y;
}

void test1_f2() {
  int x = 1;
  x = test1_f1();
  if (x == 1) {
    int *p = 0;
    *p = 3; // no-warning
  }
  if (x == 2) {
    int *p = 0;
    *p = 3; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}}
  }
}

// Test that inlining works when the declared function has less arguments
// than the actual number in the declaration.
void test2_f1() {}
int test2_f2();

void test2_f3() { 
  test2_f1(test2_f2()); // expected-warning{{too many arguments in call to 'test2_f1'}}
}

