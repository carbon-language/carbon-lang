void test(int i, float f) {
  // unqualified lookup
  f0(&i);
  f0(&f);

  // qualified lookup into the translation unit
  ::f0(&i);
  ::f0(&f);
}

// RUN: %clang_cc1 -emit-module -x c++ -verify -o %t_lookup_left.h.pch %S/Inputs/lookup_left.hpp
// RUN: %clang_cc1 -emit-module -x c++ -o %t_lookup_right.h.pch %S/Inputs/lookup_right.hpp
// RUN: %clang_cc1 -x c++ -import-module %t_lookup_left.h.pch -import-module %t_lookup_right.h.pch -verify %s
// RUN: %clang_cc1 -ast-print -x c++ -import-module %t_lookup_left.h.pch -import-module %t_lookup_right.h.pch %s | FileCheck -check-prefix=CHECK-PRINT %s

// CHECK-PRINT: int *f0(int *);
// CHECK-PRINT: float *f0(float *);
// CHECK-PRINT: void test(int i, float f)

