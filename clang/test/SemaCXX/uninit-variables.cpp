// RUN: %clang_cc1 -fsyntax-only -Wuninitialized-experimental -fsyntax-only %s -verify

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
  unsigned x; // expected-note{{declared here}} expected-note{{add initialization}}
  const bool flag = false;
  if (flag && (x = test3_aux()) == 0) {
    x = 1;
  }
  return x; // expected-warning{{variable 'x' is possibly uninitialized when used here}}
}

enum test4_A {
 test4_A_a, test_4_A_b
};
test4_A test4() {
 test4_A a; // expected-note{{variable 'a' is declared here}}
 return a; // expected-warning{{variable 'a' is possibly uninitialized when used here}}
}

