// RUN: %clang_cc1 -fsyntax-only -Wuninitialized-experimental -fsyntax-only %s -verify

int test1() {
  int x;
  return x; // expected-warning{{use of uninitialized variable 'x'}}
}

int test2() {
  int x = 0;
  return x; // no-warning
}

int test3() {
  int x;
  x = 0;
  return x; // no-warning
}

int test4() {
  int x;
  ++x; // expected-warning{{use of uninitialized variable 'x'}}
  return x; 
}

int test5() {
  int x, y;
  x = y; // expected-warning{{use of uninitialized variable 'y'}}
  return x;
}

int test6() {
  int x;
  x += 2; // expected-warning{{use of uninitialized variable 'x'}}
  return x;
}

int test7(int y) {
  int x;
  if (y)
    x = 1;
  return x;  // expected-warning{{use of uninitialized variable 'x'}}
}

int test8(int y) {
  int x;
  if (y)
    x = 1;
  else
    x = 0;
  return x; // no-warning
}

int test9(int n) {
  int x;
  for (unsigned i = 0 ; i < n; ++i) {
    if (i == n - 1)
      break;
    x = 1;    
  }
  return x; // expected-warning{{use of uninitialized variable 'x'}}
}

int test10(unsigned n) {
  int x;
  for (unsigned i = 0 ; i < n; ++i) {
    x = 1;
  }
  return x; // expected-warning{{use of uninitialized variable 'x'}}
}

int test11(unsigned n) {
  int x;
  for (unsigned i = 0 ; i <= n; ++i) {
    x = 1;
  }
  return x; // expected-warning{{use of uninitialized variable 'x'}}
}

void test12(unsigned n) {
  for (unsigned i ; n ; ++i) ; // expected-warning{{use of uninitialized variable 'i'}}
}

int test13() {
  static int i;
  return i; // no-warning
}

// Simply don't crash on this test case.
void test14() {
  const char *p = 0;
  for (;;) {}
}

void test15() {
  int x = x; // expected-warning{{use of uninitialized variable 'x'}}
}

// Don't warn in the following example; shows dataflow confluence.
char *test16_aux();
void test16() {
  char *p = test16_aux();
  for (unsigned i = 0 ; i < 100 ; i++)
    p[i] = 'a'; // no-warning
}

void test17() {
  // Don't warn multiple times about the same uninitialized variable
  // along the same path.
  int *x;
  *x = 1; // expected-warning{{use of uninitialized variable 'x'}}
  *x = 1; // no-warning
}

int test18(int x, int y) {
  int z;
  if (x && y && (z = 1)) {
    return z; // no-warning
  }
  return 0;
}

int test19_aux1();
int test19_aux2();
int test19_aux3(int *x);
int test19() {
  int z;
  if (test19_aux1() + test19_aux2() && test19_aux1() && test19_aux3(&z))
    return z; // no-warning
  return 0;
}

int test20() {
  int z;
  if ((test19_aux1() + test19_aux2() && test19_aux1()) || test19_aux3(&z))
    return z; // expected-warning{{use of uninitialized variable 'z'}}
  return 0;
}

int test21(int x, int y) {
  int z;
  if ((x && y) || test19_aux3(&z) || test19_aux2())
    return z; // expected-warning{{use of uninitialized variable 'z'}}
  return 0;
}

int test22() {
  int z;
  while (test19_aux1() + test19_aux2() && test19_aux1() && test19_aux3(&z))
    return z; // no-warning
  return 0;
}

int test23() {
  int z;
  for ( ; test19_aux1() + test19_aux2() && test19_aux1() && test19_aux3(&z) ; )
    return z; // no-warning
  return 0;
}

// The basic uninitialized value analysis doesn't have enough path-sensitivity
// to catch initializations relying on control-dependencies spanning multiple
// conditionals.  This possibly can be handled by making the CFG itself
// represent such control-dependencies, but it is a niche case.
int test24(int flag) {
  unsigned val;
  if (flag)
    val = 1;
  if (!flag)
    val = 1;
  return val; // expected-warning{{use of uninitialized variable 'val'}}
}

