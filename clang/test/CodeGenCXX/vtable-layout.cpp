// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm-only -fdump-vtable-layouts 2>&1 | FileCheck %s

// For now, just verify this doesn't crash.
namespace test0 {
  struct Obj {};

  struct Base {           virtual const Obj *foo() = 0; };
  struct Derived : Base { virtual       Obj *foo() { return new Obj(); } };

  void test(Derived *D) { D->foo(); }
}

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

// Test that a pure virtual member doesn't get a thunk.

// CHECK:     Vtable for 'Test4::F' (5 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | Test4::F RTTI
// CHECK-NEXT:       -- (Test4::A, 0) vtable address --
// CHECK-NEXT:       -- (Test4::F, 0) vtable address --
// CHECK-NEXT:   2 | Test4::R3 *Test4::F::f() [pure]
// CHECK-NEXT:   3 | void Test4::F::g()
// CHECK-NEXT:   4 | Test4::R3 *Test4::F::f() [pure]
struct F : A {
  virtual void g();
  virtual R3 *f() = 0;
};
void F::g() { }

}

namespace Test5 {

// Simple secondary vtables without 'this' pointer adjustments.
struct A {
  virtual void f();
  virtual void g();
  int a;
};

struct B1 : A {
  virtual void f();
  int b1;
};

struct B2 : A {
  virtual void g();
  int b2;
};

// CHECK:     Vtable for 'Test5::C' (9 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | Test5::C RTTI
// CHECK-NEXT:       -- (Test5::A, 0) vtable address --
// CHECK-NEXT:       -- (Test5::B1, 0) vtable address --
// CHECK-NEXT:       -- (Test5::C, 0) vtable address --
// CHECK-NEXT:   2 | void Test5::B1::f()
// CHECK-NEXT:   3 | void Test5::A::g()
// CHECK-NEXT:   4 | void Test5::C::h()
// CHECK-NEXT:   5 | offset_to_top (-16)
// CHECK-NEXT:   6 | Test5::C RTTI
// CHECK-NEXT:       -- (Test5::A, 16) vtable address --
// CHECK-NEXT:       -- (Test5::B2, 16) vtable address --
// CHECK-NEXT:   7 | void Test5::A::f()
// CHECK-NEXT:   8 | void Test5::B2::g()
struct C : B1, B2 {
  virtual void h();
};
void C::h() { }  
}

namespace Test6 {

// Simple non-virtual 'this' pointer adjustments.
struct A1 {
  virtual void f();
  int a;
};

struct A2 {
  virtual void f();
  int a;
};

// CHECK:     Vtable for 'Test6::C' (6 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | Test6::C RTTI
// CHECK-NEXT:       -- (Test6::A1, 0) vtable address --
// CHECK-NEXT:       -- (Test6::C, 0) vtable address --
// CHECK-NEXT:   2 | void Test6::C::f()
// CHECK-NEXT:   3 | offset_to_top (-16)
// CHECK-NEXT:   4 | Test6::C RTTI
// CHECK-NEXT:       -- (Test6::A2, 16) vtable address --
// CHECK-NEXT:   5 | void Test6::C::f()
// CHECK-NEXT:       [this adjustment: -16 non-virtual]
struct C : A1, A2 {
  virtual void f();
};
void C::f() { }

}

namespace Test7 {

// Test that the D::f overrider for A::f have different 'this' pointer
// adjustments in the two A base subobjects.

struct A {
  virtual void f();
  int a;
};

struct B1 : A { };
struct B2 : A { };

struct C { virtual void c(); };

// CHECK:     Vtable for 'Test7::D' (10 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | Test7::D RTTI
// CHECK-NEXT:       -- (Test7::C, 0) vtable address --
// CHECK-NEXT:       -- (Test7::D, 0) vtable address --
// CHECK-NEXT:   2 | void Test7::C::c()
// CHECK-NEXT:   3 | void Test7::D::f()
// CHECK-NEXT:   4 | offset_to_top (-8)
// CHECK-NEXT:   5 | Test7::D RTTI
// CHECK-NEXT:       -- (Test7::A, 8) vtable address --
// CHECK-NEXT:       -- (Test7::B1, 8) vtable address --
// CHECK-NEXT:   6 | void Test7::D::f()
// CHECK-NEXT:       [this adjustment: -8 non-virtual]
// CHECK-NEXT:   7 | offset_to_top (-24)
// CHECK-NEXT:   8 | Test7::D RTTI
// CHECK-NEXT:       -- (Test7::A, 24) vtable address --
// CHECK-NEXT:       -- (Test7::B2, 24) vtable address --
// CHECK-NEXT:   9 | void Test7::D::f()
// CHECK-NEXT:       [this adjustment: -24 non-virtual]
struct D : C, B1, B2 {
  virtual void f();
};
void D::f() { }

}

namespace Test8 {

// Test that we don't try to layout vtables for classes that don't have
// virtual bases or virtual member functions.

struct A { };

// CHECK:     Vtable for 'Test8::B' (3 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | Test8::B RTTI
// CHECK-NEXT:       -- (Test8::B, 0) vtable address --
// CHECK-NEXT:   2 | void Test8::B::f()
struct B : A { 
  virtual void f();
};
void B::f() { }

}

namespace Test9 {

// Simple test of vbase offsets.

struct A1 { int a1; };
struct A2 { int a2; };

// CHECK:     Vtable for 'Test9::B' (5 entries).
// CHECK-NEXT:   0 | vbase_offset (16)
// CHECK-NEXT:   1 | vbase_offset (12)
// CHECK-NEXT:   2 | offset_to_top (0)
// CHECK-NEXT:   3 | Test9::B RTTI
// CHECK-NEXT:       -- (Test9::B, 0) vtable address --
// CHECK-NEXT:   4 | void Test9::B::f()
struct B : virtual A1, virtual A2 {
  int b;

  virtual void f();
};


void B::f() { }

}

namespace Test10 {

// Test for a bug where we would not emit secondary vtables for bases
// of a primary base.
struct A1 { virtual void a1(); };
struct A2 { virtual void a2(); };

// CHECK:     Vtable for 'Test10::C' (7 entries).
// CHECK-NEXT:   0 | offset_to_top (0)
// CHECK-NEXT:   1 | Test10::C RTTI
// CHECK-NEXT:       -- (Test10::A1, 0) vtable address --
// CHECK-NEXT:       -- (Test10::B, 0) vtable address --
// CHECK-NEXT:       -- (Test10::C, 0) vtable address --
// CHECK-NEXT:   2 | void Test10::A1::a1()
// CHECK-NEXT:   3 | void Test10::C::f()
// CHECK-NEXT:   4 | offset_to_top (-8)
// CHECK-NEXT:   5 | Test10::C RTTI
// CHECK-NEXT:       -- (Test10::A2, 8) vtable address --
// CHECK-NEXT:   6 | void Test10::A2::a2()
struct B : A1, A2 {
  int b;
};

struct C : B {
  virtual void f();
};
void C::f() { }

}

namespace Test11 {

// Very simple test of vtables for virtual bases.
struct A1 { int a; };
struct A2 { int b; };

struct B : A1, virtual A2 {
  int b;
};

// CHECK:     Vtable for 'Test11::C' (8 entries).
// CHECK-NEXT:   0 | vbase_offset (24)
// CHECK-NEXT:   1 | vbase_offset (8)
// CHECK-NEXT:   2 | offset_to_top (0)
// CHECK-NEXT:   3 | Test11::C RTTI
// CHECK-NEXT:       -- (Test11::C, 0) vtable address --
// CHECK-NEXT:   4 | void Test11::C::f()
// CHECK-NEXT:   5 | vbase_offset (16)
// CHECK-NEXT:   6 | offset_to_top (-8)
// CHECK-NEXT:   7 | Test11::C RTTI
struct C : virtual B {
  virtual void f();
};
void C::f() { }

}
