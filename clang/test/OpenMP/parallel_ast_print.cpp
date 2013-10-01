// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}


template <typename T>
T tmain(T argc, T *argv) {
  T b = argc, c, d, e, f, g;
  static T a;
#pragma omp parallel
  a=2;
#pragma omp parallel default(none), private(argc,b) firstprivate(argv) shared (d)
  foo();
  return 0;
}
// CHECK: template <typename T = int> int tmain(int argc, int *argv) {
// CHECK-NEXT: int b = argc, c, d, e, f, g;
// CHECK-NEXT: static int a;
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp parallel default(none) private(argc,b) firstprivate(argv) shared(d)
// CHECK-NEXT: foo()
// CHECK: template <typename T = float> float tmain(float argc, float *argv) {
// CHECK-NEXT: float b = argc, c, d, e, f, g;
// CHECK-NEXT: static float a;
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp parallel default(none) private(argc,b) firstprivate(argv) shared(d)
// CHECK-NEXT: foo()
// CHECK: template <typename T> T tmain(T argc, T *argv) {
// CHECK-NEXT: T b = argc, c, d, e, f, g;
// CHECK-NEXT: static T a;
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp parallel default(none) private(argc,b) firstprivate(argv) shared(d)
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
#pragma omp parallel default(none), private(argc,b) firstprivate(argv)
// CHECK-NEXT: #pragma omp parallel default(none) private(argc,b) firstprivate(argv)
  foo();
// CHECK-NEXT: foo();
  return tmain(b, &b) + tmain(x, &x);
}

#endif
