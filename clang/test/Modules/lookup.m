
// lookup_left.h: expected-note{{using}}
// lookup_right.h: expected-note{{also found}}
__import_module__ lookup_left_objc;
__import_module__ lookup_right_objc;

void test(id x) {
  [x method]; // expected-warning{{multiple methods named 'method' found}}
}

// RUN: %clang_cc1 -emit-module -x objective-c -o %T/lookup_left_objc.pcm %S/Inputs/lookup_left.h
// RUN: %clang_cc1 -emit-module -x objective-c -o %T/lookup_right_objc.pcm %S/Inputs/lookup_right.h
// RUN: %clang_cc1 -x objective-c -fmodule-cache-path %T -fdisable-module-hash -verify %s
// RUN: %clang_cc1 -ast-print -x objective-c -fmodule-cache-path %T -fdisable-module-hash %s | FileCheck -check-prefix=CHECK-PRINT %s

// CHECK-PRINT: - (int) method;
// CHECK-PRINT: - (double) method
// CHECK-PRINT: void test(id x)

