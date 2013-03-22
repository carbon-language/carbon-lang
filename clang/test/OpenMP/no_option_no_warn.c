// RUN: %clang_cc1 -verify -Wno-source-uses-openmp -o - %s
// expected-no-diagnostics

int a;
#pragma omp threadprivate(a,b)
#pragma omp parallel
