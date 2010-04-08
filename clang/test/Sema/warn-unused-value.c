// RUN: %clang_cc1 -fsyntax-only -verify -Wunused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wunused %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s

int i = 0;
int j = 0;

void foo();

// PR4806
void pr4806() {
  1,foo();          // expected-warning {{expression result unused}}

  // other
  foo();
  i;                // expected-warning {{expression result unused}}

  i,foo();          // expected-warning {{expression result unused}}
  foo(),i;          // expected-warning {{expression result unused}}

  i,j,foo();        // expected-warning {{expression result unused}}
  i,foo(),j;        // expected-warning {{expression result unused}}
  foo(),i,j;        // expected-warning {{expression result unused}}

  i++;

  i++,foo();
  foo(),i++;

  i++,j,foo();      // expected-warning {{expression result unused}}
  i++,foo(),j;      // expected-warning {{expression result unused}}
  foo(),i++,j;      // expected-warning {{expression result unused}}

  i,j++,foo();      // expected-warning {{expression result unused}}
  i,foo(),j++;      // expected-warning {{expression result unused}}
  foo(),i,j++;      // expected-warning {{expression result unused}}

  i++,j++,foo();
  i++,foo(),j++;
  foo(),i++,j++;

  {};
  ({});
  ({}),foo();
  foo(),({});

  (int)1U;          // expected-warning {{expression result unused}}
  (void)1U;

  // pointer to volatile has side effect (thus no warning)
  int* pi = &i;
  volatile int* pj = &j;
  *pi;              // expected-warning {{expression result unused}}
  *pj;
}

// Don't warn about unused '||', '&&' expressions that contain assignments.
int test_logical_foo1();
int test_logical_foo2();
int test_logical_foo3();
int test_logical_bar() {
  int x = 0;
  (x = test_logical_foo1()) ||  // no-warning
  (x = test_logical_foo2()) ||  // no-warning
  (x = test_logical_foo3());    // no-warning
  return x;
}

