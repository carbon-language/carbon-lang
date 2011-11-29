
#define import __import_module__
import lookup_left_cxx;
#define IMPORT(X) __import_module__ X
IMPORT(lookup_right_cxx);

void test(int i, float f) {
  // unqualified lookup
  f0(&i);
  f0(&f);

  // qualified lookup into the translation unit
  ::f0(&i);
  ::f0(&f);
}

// RUN: rm -rf %t
// RUN: %clang_cc1 -emit-module -fmodule-cache-path %t -fmodule-name=lookup_left_cxx -x c++ %S/Inputs/module.map -verify
// RUN: %clang_cc1 -emit-module -fmodule-cache-path %t -fmodule-name=lookup_right_cxx -x c++ %S/Inputs/module.map -verify
// RUN: %clang_cc1 -x c++ -fmodule-cache-path %t %s -verify
// RUN: %clang_cc1 -ast-print -x c++ -fmodule-cache-path %t %s | FileCheck -check-prefix=CHECK-PRINT %s

// CHECK-PRINT: int *f0(int *);
// CHECK-PRINT: float *f0(float *);
// CHECK-PRINT: void test(int i, float f)

