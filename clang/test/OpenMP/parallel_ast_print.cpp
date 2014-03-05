// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}


template <typename T, int C>
T tmain(T argc, T *argv) {
  T b = argc, c, d, e, f, g;
  static T a;
#pragma omp parallel
  a=2;
#pragma omp parallel default(none), private(argc,b) firstprivate(argv) shared (d) if (argc > 0)
  foo();
#pragma omp parallel if (C)
  foo();
  return 0;
}
// CHECK: template <typename T = int, int C = 2> int tmain(int argc, int *argv) {
// CHECK-NEXT: int b = argc, c, d, e, f, g;
// CHECK-NEXT: static int a;
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp parallel default(none) private(argc,b) firstprivate(argv) shared(d) if(argc > 0)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp parallel if(2)
// CHECK-NEXT: foo()
// CHECK: template <typename T = float, int C = 0> float tmain(float argc, float *argv) {
// CHECK-NEXT: float b = argc, c, d, e, f, g;
// CHECK-NEXT: static float a;
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp parallel default(none) private(argc,b) firstprivate(argv) shared(d) if(argc > 0)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp parallel if(0)
// CHECK-NEXT: foo()
// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T b = argc, c, d, e, f, g;
// CHECK-NEXT: static T a;
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp parallel default(none) private(argc,b) firstprivate(argv) shared(d) if(argc > 0)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp parallel if(C)
// CHECK-NEXT: foo()

int main (int argc, char **argv) {
  float x;
  int b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
#pragma omp parallel
// CHECK-NEXT: #pragma omp parallel
  a=2;
// CHECK-NEXT: a = 2;
#pragma omp parallel default(none), private(argc,b) firstprivate(argv) if (argc > 0)
// CHECK-NEXT: #pragma omp parallel default(none) private(argc,b) firstprivate(argv) if(argc > 0)
  foo();
// CHECK-NEXT: foo();
  return tmain<int, 2>(b, &b) + tmain<float, 0>(x, &x);
}

#endif
