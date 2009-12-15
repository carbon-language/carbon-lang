// RUN: clang -cc1 %s -emit-llvm -o - | FileCheck %s

struct A { virtual void a(); };
struct B : A {};
struct C : B { virtual void a(); };
void (C::*x)() = &C::a;

// CHECK: @x = global %0 { i{{[0-9]+}} 1, i{{[0-9]+}} 0 }
