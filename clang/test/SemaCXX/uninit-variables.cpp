// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -fsyntax-only %s -verify

int test1_aux(int &x);
int test1() {
  int x;
  test1_aux(x);
  return x; // no-warning
}

int test2_aux() {
  int x;
  int &y = x;
  return x; // no-warning
}

// Handle cases where the CFG may constant fold some branches, thus
// mitigating the need for some path-sensitivity in the analysis.
unsigned test3_aux();
unsigned test3() {
  unsigned x = 0;
  const bool flag = true;
  if (flag && (x = test3_aux()) == 0) {
    return x;
  }
  return x;
}
unsigned test3_b() {
  unsigned x ;
  const bool flag = true;
  if (flag && (x = test3_aux()) == 0) {
    x = 1;
  }
  return x; // no-warning
}
unsigned test3_c() {
  unsigned x ; // expected-warning{{use of uninitialized variable 'x'}} expected-note{{add initialization to silence this warning}}
  const bool flag = false;
  if (flag && (x = test3_aux()) == 0) {
    x = 1;
  }
  return x; // expected-note{{variable 'x' is possibly uninitialized when used here}}
}

