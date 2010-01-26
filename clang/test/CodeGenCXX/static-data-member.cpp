// RUN: %clang_cc1 -emit-llvm -o - %s

// CHECK: @_ZN1A1aE = constant i32 10

// PR5564.
struct A {
  static const int a = 10;
};

const int A::a;

struct S { 
  static int i;
};

void f() { 
  int a = S::i;
}
