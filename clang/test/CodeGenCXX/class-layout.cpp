// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// An extra byte shoudl be allocated for an empty class.
// CHECK: %struct.A = type { i8 }
struct A { } a;

// No need to add tail padding here.
// CHECK: %struct.B = type { i8*, i32 }
struct B { void *a; int b; } b;
