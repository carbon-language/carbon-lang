// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}
int g_ind = 1;
template<class T, class N> T reduct(T* arr, N num) {
  N i;
  N ind;
  N myind;
  T sum = (T)0;
// CHECK: T sum = (T)0;
#pragma omp simd private(myind, g_ind)
// CHECK-NEXT: #pragma omp simd private(myind,g_ind)
  for (i = 0; i < num; ++i) {
    myind = ind;
    T cur = arr[myind];
    ind += g_ind;
    sum += cur;
  }
}

template<class T> struct S {
  S(const T &a)
    :m_a(a)
  {}
  T result(T *v) const {
    T res;
    T val;
// CHECK: T res;
// CHECK: T val;
#pragma omp simd private(val)
// CHECK-NEXT: #pragma omp simd private(val)
    for (T i = 7; i < m_a; ++i) {
      val = v[i-7] + m_a;
      res = val;
    }
    return res;
  }
  ~S()
  {}
  T m_a;
};

int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  int k1=0,k2=0;
  static int *a;
// CHECK: static int *a;
#pragma omp simd
// CHECK-NEXT: #pragma omp simd
  for (int i=0; i < 2; ++i)*a=2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: *a = 2;
#pragma omp simd private(argc, b)
  for (int i = 0; i < 10; ++i)
  for (int j = 0; j < 10; ++j) {foo(); k1 += 8; k2 += 8;}
// CHECK-NEXT: #pragma omp simd private(argc,b)
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: for (int j = 0; j < 10; ++j) {
// CHECK-NEXT: foo();
// CHECK-NEXT: k1 += 8;
// CHECK-NEXT: k2 += 8;
// CHECK-NEXT: }
  for (int i = 0; i < 10; ++i)foo();
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
#pragma omp simd
// CHECK-NEXT: #pragma omp simd
  for (int i = 0; i < 10; ++i)foo();
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
  return (0);
}

#endif
