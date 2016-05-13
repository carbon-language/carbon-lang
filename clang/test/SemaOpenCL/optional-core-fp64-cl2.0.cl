// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0
// expected-no-diagnostics

void f1(double da) {
  double d;
  (void) 1.0;
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

void f2(void) {
  double d;
  (void) 1.0;
}

#pragma OPENCL EXTENSION cl_khr_fp64 : disable

void f3(void) {
  double d;
}
