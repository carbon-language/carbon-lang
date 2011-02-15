// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

void f1(double da) { // expected-error {{requires cl_khr_fp64 extension}}
  double d; // expected-error {{requires cl_khr_fp64 extension}}
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

void f2(void) {
  double d;
}

#pragma OPENCL EXTENSION cl_khr_fp64 : disable

void f3(void) {
  double d; // expected-error {{requires cl_khr_fp64 extension}}
}
