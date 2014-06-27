// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  static T a;
// CHECK: static T a;
#pragma omp parallel private(g)
#pragma omp single private(argc, b), firstprivate(c, d), nowait copyprivate(g)
  foo();
  // CHECK-NEXT: #pragma omp parallel private(g)
  // CHECK-NEXT: #pragma omp single private(argc,b) firstprivate(c,d) nowait copyprivate(g)
  // CHECK-NEXT: foo();
  return T();
}

int main(int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
#pragma omp parallel private(g)
#pragma omp single private(argc, b), firstprivate(argv, c), nowait copyprivate(g)
  foo();
  // CHECK-NEXT: #pragma omp parallel private(g)
  // CHECK-NEXT: #pragma omp single private(argc,b) firstprivate(argv,c) nowait copyprivate(g)
  // CHECK-NEXT: foo();
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
