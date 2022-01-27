// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp master
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

int foo();

int main() {
  #pragma omp master
  ;
  #pragma omp master nowait // expected-error {{unexpected OpenMP clause 'nowait' in directive '#pragma omp master'}}
  #pragma omp master unknown // expected-warning {{extra tokens at the end of '#pragma omp master' are ignored}}
  foo();
  {
    #pragma omp master
  } // expected-error {{expected statement}}
  #pragma omp for
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp master // expected-error {{region cannot be closely nested inside 'for' region}}
    foo();
  }
  #pragma omp sections
  {
    foo();
    #pragma omp master // expected-error {{region cannot be closely nested inside 'sections' region}}
    foo();
  }
  #pragma omp single
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp master allocate(i) // expected-error {{region cannot be closely nested inside 'single' region}} expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp master'}}
    foo();
  }
  #pragma omp master
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp master
    foo();
  }
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i)
  #pragma omp master // expected-error {{region cannot be closely nested inside 'for' region}}
  {
    foo();
  }

  return 0;
}

int foo() {
  L1: // expected-note {{jump exits scope of OpenMP structured block}}
    foo();
  #pragma omp master
  {
    foo();
    goto L1; // expected-error {{cannot jump from this goto statement to its label}}
  }
  goto L2; // expected-error {{cannot jump from this goto statement to its label}}
  #pragma omp master
  { // expected-note {{jump bypasses OpenMP structured block}}
    L2:
    foo();
  }

  return 0;
}
