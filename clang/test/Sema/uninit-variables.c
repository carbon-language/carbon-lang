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
  
