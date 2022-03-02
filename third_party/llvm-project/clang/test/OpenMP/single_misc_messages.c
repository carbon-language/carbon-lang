// RUN: %clang_cc1 -fsyntax-only -fopenmp -verify %s -Wuninitialized

// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -verify %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp single
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

void foo(void);

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp single'}}
#pragma omp single

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp single'}}
#pragma omp single foo

void test_no_clause(void) {
  int i;
#pragma omp single
  foo();

#pragma omp single
  ++i;
}

void test_branch_protected_scope(void) {
  int i = 0;
L1:
  ++i;

  int x[24];

#pragma omp parallel
#pragma omp single
  {
    if (i == 5)
      goto L1; // expected-error {{use of undeclared label 'L1'}}
    else if (i == 6)
      return; // expected-error {{cannot return from OpenMP region}}
    else if (i == 7)
      goto L2;
    else if (i == 8) {
    L2:
      x[i]++;
    }
  }

  if (x[0] == 0)
    goto L2; // expected-error {{use of undeclared label 'L2'}}
  else if (x[1] == 1)
    goto L1;
}

void test_invalid_clause(void) {
  int i;
#pragma omp parallel
// expected-warning@+1 {{extra tokens at the end of '#pragma omp single' are ignored}}
#pragma omp single foo bar
  foo();
}

void test_non_identifiers(void) {
  int i, x;

#pragma omp parallel
// expected-warning@+1 {{extra tokens at the end of '#pragma omp single' are ignored}}
#pragma omp single;
  foo();
#pragma omp parallel
// expected-error@+2 {{unexpected OpenMP clause 'linear' in directive '#pragma omp single'}}
// expected-warning@+1 {{extra tokens at the end of '#pragma omp single' are ignored}}
#pragma omp single linear(x);
  foo();

#pragma omp parallel
// expected-warning@+1 {{extra tokens at the end of '#pragma omp single' are ignored}}
#pragma omp single private(x);
  foo();

#pragma omp parallel
// expected-warning@+1 {{extra tokens at the end of '#pragma omp single' are ignored}}
#pragma omp single, private(x);
  foo();
}

void test_private(void) {
  int i;
#pragma omp parallel
// expected-error@+2 {{expected expression}}
// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp single private(
  foo();
#pragma omp parallel
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp single private(,
  foo();
#pragma omp parallel
// expected-error@+1 2 {{expected expression}}
#pragma omp single private(, )
  foo();
#pragma omp parallel
// expected-error@+1 {{expected expression}}
#pragma omp single private()
  foo();
#pragma omp parallel
// expected-error@+1 {{expected expression}}
#pragma omp single private(int)
  foo();
#pragma omp parallel
// expected-error@+1 {{expected variable name}}
#pragma omp single private(0)
  foo();

  int x, y, z;
#pragma omp parallel
#pragma omp single private(x)
  foo();
#pragma omp parallel
#pragma omp single private(x, y)
  foo();
#pragma omp parallel
#pragma omp single private(x, y, z)
  foo();
}

void test_firstprivate(void) {
  int i;
#pragma omp parallel
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected expression}}
#pragma omp single firstprivate(
  foo();

#pragma omp parallel
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp single firstprivate(,
  foo();
#pragma omp parallel
// expected-error@+1 2 {{expected expression}}
#pragma omp single firstprivate(, )
  foo();
#pragma omp parallel
// expected-error@+1 {{expected expression}}
#pragma omp single firstprivate()
  foo();
#pragma omp parallel
// expected-error@+1 {{expected expression}}
#pragma omp single firstprivate(int)
  foo();
#pragma omp parallel
// expected-error@+1 {{expected variable name}}
#pragma omp single firstprivate(0)
  foo();
}

void test_nowait(void) {
#pragma omp single nowait nowait // expected-error {{directive '#pragma omp single' cannot contain more than one 'nowait' clause}}
  for (int i = 0; i < 16; ++i)
    ;
}
