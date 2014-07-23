// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp=libiomp5 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <class T>
T foo(T arg) {
  T a;
#pragma omp atomic
  a++;
#pragma omp atomic read
  a = arg;
  return T();
}

// CHECK: int a;
// CHECK-NEXT: #pragma omp atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = arg;
// CHECK: T a;
// CHECK-NEXT: #pragma omp atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = arg;

int main(int argc, char **argv) {
  int a;
// CHECK: int a;
#pragma omp atomic
  a++;
#pragma omp atomic read
  a = argc;
  // CHECK-NEXT: #pragma omp atomic
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read
  // CHECK-NEXT: a = argc;
  return foo(a);
}

#endif
