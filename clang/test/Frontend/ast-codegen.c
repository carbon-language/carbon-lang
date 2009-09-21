// RUN: clang -emit-ast -o %t.ast %s &&
// RUN: clang -emit-llvm -S -o - %t.ast | FileCheck %s

// CHECK: module asm "foo"
__asm__("foo");

// CHECK: @g0 = common global i32 0, align 4
int g0;

// CHECK: define i32 @f0()
int f0() {
}
