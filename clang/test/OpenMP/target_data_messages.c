// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

void foo() { }

int main(int argc, char **argv) {
  L1:
    foo();
  #pragma omp target data
  {
    foo();
    goto L1; // expected-error {{use of undeclared label 'L1'}}
  }
  goto L2; // expected-error {{use of undeclared label 'L2'}}
  #pragma omp target data
  L2:
  foo();

  #pragma omp target data(i) // expected-warning {{extra tokens at the end of '#pragma omp target data' are ignored}}
  {
    foo();
  }
  #pragma omp target unknown // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  {
    foo();
  }
  return 0;
}
