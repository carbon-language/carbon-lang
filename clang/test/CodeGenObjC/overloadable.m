// rdar://6657613
// RUN: %clang_cc1 -cxx-abi itanium -emit-llvm %s -o - | FileCheck %s

@class C;

// CHECK: _Z1fP11objc_object
// CHECK-NOT: _Z1fP11objc_object
void __attribute__((overloadable)) f(id c) { }

// CHECK: _Z1fP1C
// CHECK-NOT: _Z1fP1C
void __attribute__((overloadable)) f(C *c) { }
