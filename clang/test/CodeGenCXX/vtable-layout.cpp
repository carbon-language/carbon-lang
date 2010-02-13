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
// CHECK-NEXT:      -- (Test3::A, 0) vtable address --
// CHECK-NEXT:      -- (Test3::B, 0) vtable address --
// CHECK-NEXT:  2 | void Test3::B::f()
// CHECK-NEXT:  3 | void Test3::B::g()
struct B : A {
  virtual void f();
  virtual void g();
};
void B::f() { }

// CHECK:     Vtable for 'Test3::C' (5 entries).
// CHECK-NEXT:  0 | offset_to_top (0)
// CHECK-NEXT:  1 | Test3::C RTTI
// CHECK-NEXT:     -- (Test3::A, 0) vtable address --
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
// CHECK-NEXT:     -- (Test3::A, 0) vtable address --
// CHECK-NEXT:     -- (Test3::B, 0) vtable address --
// CHECK-NEXT:     -- (Test3::D, 0) vtable address --
// CHECK-NEXT:  2 | void Test3::D::f()
// CHECK-NEXT:  3 | void Test3::D::g()
// CHECK-NEXT:  4 | void Test3::D::h()
struct D : B {
  virtual void f();
  virtual void g();
  virtual void h();
};

void D::f() { } 
}

namespace Test4 {

// Test non-virtual result adjustments.

struct R1 { int r1; };
struct R2 { int r2; };
struct R3 : R1, R2 { int r3; };

struct A {
  virtual R2 *f();
};

// CHECK:     Vtable for 'Test4::B' (4 entries).
// CHECK-NEXT:  0 | offset_to_top (0)
// CHECK-NEXT:  1 | Test4::B RTTI
// CHECK-NEXT:      -- (Test4::A, 0) vtable address --
// CHECK-NEXT:      -- (Test4::B, 0) vtable address --
// CHECK-NEXT:  2 | Test4::R3 *Test4::B::f()
// CHECK-NEXT:      [return adjustment: 4 non-virtual]
// CHECK-NEXT:  3 | Test4::R3 *Test4::B::f()

struct B : A {
  virtual R3 *f();
};
R3 *B::f() { return 0; }

// Test virtual result adjustments.
struct V1 { int v1; };
struct V2 : virtual V1 { int v1; };

struct C {
  virtual V1 *f(); 
};

// CHECK:     Vtable for 'Test4::D' (4 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | Test4::D RTTI
// CHECK-NEXT:       -- (Test4::C, 0) vtable address --
// CHECK-NEXT:       -- (Test4::D, 0) vtable address --
// CHECK-NEXT:   2 | Test4::V2 *Test4::D::f()
// CHECK-NEXT:       [return adjustment: 0 non-virtual, -24 vbase offset offset]
// CHECK-NEXT:   3 | Test4::V2 *Test4::D::f()
struct D : C {
  virtual V2 *f();
};
V2 *D::f() { return 0; };

// Virtual result adjustments with an additional non-virtual adjustment.
struct V3 : virtual R3 { int r3; };

// CHECK:     Vtable for 'Test4::E' (4 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | Test4::E RTTI
// CHECK-NEXT:       -- (Test4::A, 0) vtable address --
// CHECK-NEXT:       -- (Test4::E, 0) vtable address --
// CHECK-NEXT:   2 | Test4::V3 *Test4::E::f()
// CHECK-NEXT:       [return adjustment: 4 non-virtual, -24 vbase offset offset]
// CHECK-NEXT:   3 | Test4::V3 *Test4::E::f()

struct E : A {
  virtual V3 *f();
};
V3 *E::f() { return 0;}

}

// For now, just verify this doesn't crash.
namespace test0 {
  struct Obj {};

  struct Base {           virtual const Obj *foo() = 0; };
  struct Derived : Base { virtual       Obj *foo() { return new Obj(); } };

  void test(Derived *D) { D->foo(); }
}
