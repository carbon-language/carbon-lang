// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s

#pragma omp // expected-error {{expected an OpenMP directive}}
#pragma omp unknown_directive // expected-error {{expected an OpenMP directive}}

void foo() {
#pragma omp // expected-error {{expected an OpenMP directive}}
#pragma omp unknown_directive // expected-error {{expected an OpenMP directive}}
}
