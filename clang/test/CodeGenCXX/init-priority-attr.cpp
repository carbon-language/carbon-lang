// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -O2 -emit-llvm -o - | FileCheck %s
// PR11480

void foo(int);

class A {
public:
  A() { foo(1); }
};

class A1 {
public:
  A1() { foo(2); }
};

class B {
public:
  B() { foo(3); }
};

class C {
public:
  static A a;
  C() { foo(4); }
};


A C::a = A();

// CHECK: @llvm.global_ctors = appending global [3 x { i32, void ()* }] [{ i32, void ()* } { i32 200, void ()* @_GLOBAL__I_000200 }, { i32, void ()* } { i32 300, void ()* @_GLOBAL__I_000300 }, { i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

// CHECK: _GLOBAL__I_000200()
// CHECK: _Z3fooi(i32 3)
// CHECK-NEXT: ret void

// CHECK: _GLOBAL__I_000300()
// CHECK: _Z3fooi(i32 2)
// CHECK-NEXT: _Z3fooi(i32 1)
// CHECK-NEXT: ret void

// CHECK: _GLOBAL__I_a()
// CHECK: _Z3fooi(i32 1)
// CHECK-NEXT: _Z3fooi(i32 4)
// CHECK-NEXT: ret void

C c;
A1 a1 __attribute__((init_priority (300)));
A a __attribute__((init_priority (300)));
B b __attribute__((init_priority (200)));
