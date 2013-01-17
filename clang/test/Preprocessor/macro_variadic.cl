// RUN: %clang_cc1 -verify %s

#define X(...) 1 // expected-error {{variadic macros not supported in OpenCL}}
