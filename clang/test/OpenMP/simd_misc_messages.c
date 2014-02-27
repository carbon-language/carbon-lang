// RUN: %clang_cc1 -fsyntax-only -fopenmp -verify %s

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp simd'}}
#pragma omp simd

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp simd'}}
#pragma omp simd foo

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp simd'}}
#pragma omp simd safelen(4)

void test_no_clause()
{
  int i;
  #pragma omp simd
  for (i = 0; i < 16; ++i) ;

  // expected-error@+2 {{statement after '#pragma omp simd' must be a for loop}}
  #pragma omp simd
  ++i;
}

void test_branch_protected_scope()
{
  int i = 0;
L1:
  ++i;

  int x[24];

  #pragma omp simd
  for (i = 0; i < 16; ++i) {
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

void test_invalid_clause()
{
  int i;
  // expected-warning@+1 {{extra tokens at the end of '#pragma omp simd' are ignored}}
  #pragma omp simd foo bar
  for (i = 0; i < 16; ++i) ;
}

void test_non_identifiers()
{
  int i, x;
  // expected-warning@+1 {{extra tokens at the end of '#pragma omp simd' are ignored}}
  #pragma omp simd;
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{unexpected OpenMP clause 'firstprivate' in directive '#pragma omp simd'}}
  // expected-warning@+1 {{extra tokens at the end of '#pragma omp simd' are ignored}}
  #pragma omp simd firstprivate(x);
  for (i = 0; i < 16; ++i) ;
  // expected-warning@+1 {{extra tokens at the end of '#pragma omp simd' are ignored}}
  #pragma omp simd , private(x);
  for (i = 0; i < 16; ++i) ;
}

