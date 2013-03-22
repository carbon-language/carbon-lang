// RUN: %clang_cc1 -verify -Wsource-uses-openmp -o - %s

int a;
#pragma omp threadprivate(a,b) // expected-warning {{unexpected '#pragma omp ...' in program}}
#pragma omp parallel
