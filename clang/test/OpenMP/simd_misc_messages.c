// RUN: %clang_cc1 -fsyntax-only -fopenmp=libiomp5 -verify %s

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
  #pragma omp simd private(x);
  for (i = 0; i < 16; ++i) ;

  // expected-warning@+1 {{extra tokens at the end of '#pragma omp simd' are ignored}}
  #pragma omp simd , private(x);
  for (i = 0; i < 16; ++i) ;
}

extern int foo();
void test_safelen()
{
  int i;
  // expected-error@+1 {{expected '('}}
  #pragma omp simd safelen
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp simd safelen(
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}}
  #pragma omp simd safelen()
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp simd safelen(,
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}}  expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp simd safelen(,)
  for (i = 0; i < 16; ++i) ;
  // expected-warning@+2 {{extra tokens at the end of '#pragma omp simd' are ignored}}
  // expected-error@+1 {{expected '('}}
  #pragma omp simd safelen 4)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp simd safelen(4
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp simd safelen(4,
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp simd safelen(4,)
  for (i = 0; i < 16; ++i) ;
  // xxpected-error@+1 {{expected expression}}
  #pragma omp simd safelen(4)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp simd safelen(4 4)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp simd safelen(4,,4)
  for (i = 0; i < 16; ++i) ;
  #pragma omp simd safelen(4)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp simd safelen(4,8)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expression is not an integer constant expression}}
  #pragma omp simd safelen(2.5)
  for (i = 0; i < 16; ++i);
  // expected-error@+1 {{expression is not an integer constant expression}}
  #pragma omp simd safelen(foo())
  for (i = 0; i < 16; ++i);
  // expected-error@+1 {{argument to 'safelen' clause must be a positive integer value}}
  #pragma omp simd safelen(-5)
  for (i = 0; i < 16; ++i);
  // expected-error@+1 {{argument to 'safelen' clause must be a positive integer value}}
  #pragma omp simd safelen(0)
  for (i = 0; i < 16; ++i);
  // expected-error@+1 {{argument to 'safelen' clause must be a positive integer value}}
  #pragma omp simd safelen(5-5)
  for (i = 0; i < 16; ++i);
}

void test_collapse()
{
  int i;
  // expected-error@+1 {{expected '('}}
  #pragma omp simd collapse
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp simd collapse(
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}}
  #pragma omp simd collapse()
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp simd collapse(,
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}}  expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp simd collapse(,)
  for (i = 0; i < 16; ++i) ;
  // expected-warning@+2 {{extra tokens at the end of '#pragma omp simd' are ignored}}
  // expected-error@+1 {{expected '('}}
  #pragma omp simd collapse 4)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp simd collapse(4
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp simd collapse(4,
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp simd collapse(4,)
  for (i = 0; i < 16; ++i) ;
  // xxpected-error@+1 {{expected expression}}
  #pragma omp simd collapse(4)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp simd collapse(4 4)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp simd collapse(4,,4)
  for (i = 0; i < 16; ++i) ;
  #pragma omp simd collapse(4)
  for (int i1 = 0; i1 < 16; ++i1)
    for (int i2 = 0; i2 < 16; ++i2)
      for (int i3 = 0; i3 < 16; ++i3)
        for (int i4 = 0; i4 < 16; ++i4)
          foo();
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp simd collapse(4,8)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expression is not an integer constant expression}}
  #pragma omp simd collapse(2.5)
  for (i = 0; i < 16; ++i);
  // expected-error@+1 {{expression is not an integer constant expression}}
  #pragma omp simd collapse(foo())
  for (i = 0; i < 16; ++i);
  // expected-error@+1 {{argument to 'collapse' clause must be a positive integer value}}
  #pragma omp simd collapse(-5)
  for (i = 0; i < 16; ++i);
  // expected-error@+1 {{argument to 'collapse' clause must be a positive integer value}}
  #pragma omp simd collapse(0)
  for (i = 0; i < 16; ++i);
  // expected-error@+1 {{argument to 'collapse' clause must be a positive integer value}}
  #pragma omp simd collapse(5-5)
  for (i = 0; i < 16; ++i);
}

void test_linear()
{
  int i;
  // expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp simd linear(
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp simd linear(,
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected expression}}
  #pragma omp simd linear(,)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}}
  #pragma omp simd linear()
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}}
  #pragma omp simd linear(int)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected variable name}}
  #pragma omp simd linear(0)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{use of undeclared identifier 'x'}}
  #pragma omp simd linear(x)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{use of undeclared identifier 'x'}}
  // expected-error@+1 {{use of undeclared identifier 'y'}}
  #pragma omp simd linear(x, y)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+3 {{use of undeclared identifier 'x'}}
  // expected-error@+2 {{use of undeclared identifier 'y'}}
  // expected-error@+1 {{use of undeclared identifier 'z'}}
  #pragma omp simd linear(x, y, z)
  for (i = 0; i < 16; ++i) ;

  int x, y;
  // expected-error@+1 {{expected expression}}
  #pragma omp simd linear(x:)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp simd linear(x:,)
  for (i = 0; i < 16; ++i) ;
  #pragma omp simd linear(x:1)
  for (i = 0; i < 16; ++i) ;
  #pragma omp simd linear(x:2*2)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp simd linear(x:1,y)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp simd linear(x:1,y,z:1)
  for (i = 0; i < 16; ++i) ;

  // expected-note@+2 {{defined as linear}}
  // expected-error@+1 {{linear variable cannot be linear}}
  #pragma omp simd linear(x) linear(x)
  for (i = 0; i < 16; ++i) ;

  // expected-note@+2 {{defined as private}}
  // expected-error@+1 {{private variable cannot be linear}}
  #pragma omp simd private(x) linear(x)
  for (i = 0; i < 16; ++i) ;

  // expected-note@+2 {{defined as linear}}
  // expected-error@+1 {{linear variable cannot be private}}
  #pragma omp simd linear(x) private(x)
  for (i = 0; i < 16; ++i) ;

  // expected-warning@+1 {{zero linear step (x and other variables in clause should probably be const)}}
  #pragma omp simd linear(x,y:0)
  for (i = 0; i < 16; ++i) ;
}

void test_private()
{
  int i;
  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp simd private(
  for (i = 0; i < 16; ++i) ;
  // expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
  // expected-error@+1 2 {{expected expression}}
  #pragma omp simd private(,
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 2 {{expected expression}}
  #pragma omp simd private(,)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}}
  #pragma omp simd private()
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected expression}}
  #pragma omp simd private(int)
  for (i = 0; i < 16; ++i) ;
  // expected-error@+1 {{expected variable name}}
  #pragma omp simd private(0)
  for (i = 0; i < 16; ++i) ;

  int x, y, z;
  #pragma omp simd private(x)
  for (i = 0; i < 16; ++i) ;
  #pragma omp simd private(x, y)
  for (i = 0; i < 16; ++i) ;
  #pragma omp simd private(x, y, z)
  for (i = 0; i < 16; ++i) {
    x = y * i + z;
  }
}

void test_firstprivate()
{
  int i;
  // expected-error@+3 {{expected ')'}} expected-note@+3 {{to match this '('}}
  // expected-error@+2 {{unexpected OpenMP clause 'firstprivate' in directive '#pragma omp simd'}}
  // expected-error@+1 {{expected expression}}
  #pragma omp simd firstprivate(
  for (i = 0; i < 16; ++i) ;
}

