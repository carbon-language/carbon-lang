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

namespace Test3 {

// If a function in a derived class overrides a function in a primary base,
// then the function should not have an entry in the derived class (unless the return
// value requires adjusting).

// CHECK:      Vtable for 'Test3::A' (3 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | Test3::A RTTI
// CHECK-NEXT:       -- (Test3::A, 0) vtable address --
// CHECK-NEXT:   2 | void Test3::A::f()
struct A {
  virtual void f();
};
void A::f() { } 

// CHECK:     Vtable for 'Test3::B' (4 entries).
// CHECK-NEXT:  0 | offset_to_top (0)
// CHECK-NEXT:  1 | Test3::B RTTI
// CHECK-NEXT:      -- (Test3::B, 0) vtable address --
// CHECK-NEXT:  2 | void Test3::A::f()
// CHECK-NEXT:  3 | void Test3::B::g()
struct B : A {
  virtual void f();
  virtual void g();
};
void B::f() { }

// CHECK:     Vtable for 'Test3::C' (5 entries).
// CHECK-NEXT:  0 | offset_to_top (0)
// CHECK-NEXT:  1 | Test3::C RTTI
// CHECK-NEXT:     -- (Test3::C, 0) vtable address --
// CHECK-NEXT:  2 | void Test3::A::f()
// CHECK-NEXT:  3 | void Test3::C::g()
// CHECK-NEXT:  4 | void Test3::C::h()
struct C : A {
  virtual void g();
  virtual void h();
};
void C::g() { }

// CHECK:     Vtable for 'Test3::D' (5 entries).
// CHECK-NEXT:  0 | offset_to_top (0)
// CHECK-NEXT:  1 | Test3::D RTTI
// CHECK-NEXT:     -- (Test3::D, 0) vtable address --
// CHECK-NEXT:  2 | void Test3::A::f()
// CHECK-NEXT:  3 | void Test3::B::g()
// CHECK-NEXT:  4 | void Test3::D::h()
struct D : B {
  virtual void f();
  virtual void g();
  virtual void h();
};

void D::f() { } 
}
