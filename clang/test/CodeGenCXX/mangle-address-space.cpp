// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// CHECK: define void @_Z2f0Pc
void f0(char *p) { }
// CHECK: define void @_Z2f0PU3AS1c
void f0(char __attribute__((address_space(1))) *p) { }
