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
