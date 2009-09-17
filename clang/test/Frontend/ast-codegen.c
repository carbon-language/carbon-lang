// RUN: clang -emit-ast -o %t.ast %s &&
// RUN: clang -emit-llvm -S -o - %t.ast | FileCheck %s

// CHECK: module asm "foo"
__asm__("foo");
