// RUN: %clang_cc1 -ast-print -o - %s | FileCheck %s

extern "C" void f(void);
// CHECK: extern "C" void f()

extern "C" void v;
// CHECK: extern "C" void v
