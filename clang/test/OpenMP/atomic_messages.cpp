// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -ferror-limit 100 %s

int foo() {
L1:
  foo();
#pragma omp atomic
  {
    foo();
    goto L1; // expected-error {{use of undeclared label 'L1'}}
  }
  goto L2; // expected-error {{use of undeclared label 'L2'}}
#pragma omp atomic
  {
    foo();
  L2:
    foo();
  }

  return 0;
}

template <class T>
T read() {
  T a, b = 0;
// Test for atomic read
#pragma omp atomic read
// expected-error@+1 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both l-value expressions with scalar type}}
  ;
// expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'read' clause}}
#pragma omp atomic read read
  a = b;

  return T();
}

int read() {
  int a, b = 0;
// Test for atomic read
#pragma omp atomic read
// expected-error@+1 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both l-value expressions with scalar type}}
  ;
// expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'read' clause}}
#pragma omp atomic read read
  a = b;

  return read<int>();
}
