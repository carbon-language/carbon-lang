// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm-only -fdump-vtable-layouts 2>&1 | FileCheck %s
namespace Test1 {

// CHECK:      Vtable for 'Test1::A' (3 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | Test1::A RTTI
// CHECK-NEXT:       -- (Test1::A, 0) vtable address --
// CHECK-NEXT:   2 | void Test1::A::f()
struct A {
  virtual void f();
};

void A::f() { }
}

namespace Test2 {

// This is a smoke test of the vtable dumper.
// CHECK:      Vtable for 'Test2::A' (9 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | Test2::A RTTI
// CHECK-NEXT:       -- (Test2::A, 0) vtable address --
// CHECK-NEXT:   2 | void Test2::A::f()
// CHECK-NEXT:   3 | void Test2::A::f() const
// CHECK-NEXT:   4 | Test2::A *Test2::A::g(int)
// CHECK-NEXT:   5 | Test2::A::~A() [complete]
// CHECK-NEXT:   6 | Test2::A::~A() [deleting]
// CHECK-NEXT:   7 | void Test2::A::h()
// CHECK-NEXT:   8 | Test2::A &Test2::A::operator=(Test2::A const &)
struct A {
  virtual void f();
  virtual void f() const;
  
  virtual A* g(int a);
  virtual ~A();
  virtual void h();
  virtual A& operator=(const A&);
};

void A::f() { }

// Another simple vtable dumper test.
// CHECK:     Vtable for 'Test2::B' (6 entries).
// CHECK-NEXT:  0 | offset_to_top (0)
// CHECK-NEXT:  1 | Test2::B RTTI
// CHECK-NEXT:    -- (Test2::B, 0) vtable address --
// CHECK-NEXT:  2 | void Test2::B::f()
// CHECK-NEXT:  3 | void Test2::B::g() [pure]
// CHECK-NEXT:  4 | Test2::B::~B() [complete] [pure]
// CHECK-NEXT:  5 | Test2::B::~B() [deleting] [pure]

struct B {
  virtual void f();
  virtual void g() = 0;
  virtual ~B() = 0;
};

void B::f() { }

}
