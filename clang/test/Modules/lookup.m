
// lookup_left.h: expected-note{{using}}
// lookup_right.h: expected-note{{also found}}

void test(id x) {
  [x method]; // expected-warning{{multiple methods named 'method' found}}
}

// RUN: %clang_cc1 -emit-module -x objective-c -o %t_lookup_left.h.pch %S/Inputs/lookup_left.h
// RUN: %clang_cc1 -emit-module -x objective-c -o %t_lookup_right.h.pch %S/Inputs/lookup_right.h
// RUN: %clang_cc1 -x objective-c -import-module %t_lookup_left.h.pch -import-module %t_lookup_right.h.pch -verify %s
// RUN: %clang_cc1 -ast-print -x objective-c -import-module %t_lookup_left.h.pch -import-module %t_lookup_right.h.pch %s | FileCheck -check-prefix=CHECK-PRINT %s

// CHECK-PRINT: - (int) method;
// CHECK-PRINT: - (double) method
// CHECK-PRINT: void test(id x)

