// RUN: %clang_cc1 -verify -o - %s
// expected-no-diagnostics

int a;
#pragma omp threadprivate(a,b)
#pragma omp parallel
