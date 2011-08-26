
#define import __import__
import lookup_left_cxx;
#define IMPORT(X) __import__ X
IMPORT(lookup_right_cxx);

void test(int i, float f) {
  // unqualified lookup
  f0(&i);
  f0(&f);

  // qualified lookup into the translation unit
  ::f0(&i);
  ::f0(&f);
}

// RUN: %clang_cc1 -emit-module -x c++ -verify -o %T/lookup_left_cxx.pcm %S/Inputs/lookup_left.hpp
// RUN: %clang_cc1 -emit-module -x c++ -o %T/lookup_right_cxx.pcm %S/Inputs/lookup_right.hpp
// RUN: %clang_cc1 -x c++ -I %T %s -verify
// RUN: %clang_cc1 -ast-print -x c++ -I %T %s | FileCheck -check-prefix=CHECK-PRINT %s

// CHECK-PRINT: int *f0(int *);
// CHECK-PRINT: float *f0(float *);
// CHECK-PRINT: void test(int i, float f)

