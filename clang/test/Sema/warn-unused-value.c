// RUN: %clang_cc1 -fsyntax-only -verify -Wunused-value %s

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
