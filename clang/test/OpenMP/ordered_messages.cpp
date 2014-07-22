// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -ferror-limit 100 -o - %s

int foo();

template <class T>
T foo() {
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
    L1:
      foo();
    #pragma omp ordered
    {
      foo();
      goto L1; // expected-error {{use of undeclared label 'L1'}}
    }
  }
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
    foo();
    goto L2; // expected-error {{use of undeclared label 'L2'}}
    #pragma omp ordered
    {
      L2:
      foo();
    }
  }

  return T();
}

int foo() {
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
    L1:
      foo();
    #pragma omp ordered
    {
      foo();
      goto L1; // expected-error {{use of undeclared label 'L1'}}
    }
  }
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
    foo();
    goto L2; // expected-error {{use of undeclared label 'L2'}}
    #pragma omp ordered
    {
      L2:
      foo();
    }
  }

  return foo<int>();
}
