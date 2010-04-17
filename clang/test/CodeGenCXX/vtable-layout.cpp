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

namespace Test12 {

// Test that the right vcall offsets are generated in the right order.

// CHECK:      Vtable for 'Test12::B' (19 entries).
// CHECK-NEXT:    0 | vbase_offset (8)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test12::B RTTI
// CHECK-NEXT:        -- (Test12::B, 0) vtable address --
// CHECK-NEXT:    3 | void Test12::B::f()
// CHECK-NEXT:    4 | void Test12::B::a()
// CHECK-NEXT:    5 | vcall_offset (32)
// CHECK-NEXT:    6 | vcall_offset (16)
// CHECK-NEXT:    7 | vcall_offset (-8)
// CHECK-NEXT:    8 | vcall_offset (0)
// CHECK-NEXT:    9 | offset_to_top (-8)
// CHECK-NEXT:   10 | Test12::B RTTI
// CHECK-NEXT:        -- (Test12::A, 8) vtable address --
// CHECK-NEXT:        -- (Test12::A1, 8) vtable address --
// CHECK-NEXT:   11 | void Test12::A1::a1()
// CHECK-NEXT:   12 | void Test12::B::a()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]
// CHECK-NEXT:   13 | offset_to_top (-24)
// CHECK-NEXT:   14 | Test12::B RTTI
// CHECK-NEXT:        -- (Test12::A2, 24) vtable address --
// CHECK-NEXT:   15 | void Test12::A2::a2()
// CHECK-NEXT:   16 | offset_to_top (-40)
// CHECK-NEXT:   17 | Test12::B RTTI
// CHECK-NEXT:        -- (Test12::A3, 40) vtable address --
// CHECK-NEXT:   18 | void Test12::A3::a3()
struct A1 {
  virtual void a1();
  int a;
};

struct A2 {
  virtual void a2();
  int a;
};

struct A3 {
  virtual void a3();
  int a;
};

struct A : A1, A2, A3 {
  virtual void a();
  int i;
};

struct B : virtual A {
  virtual void f();

  virtual void a();
};
void B::f() { } 

}

namespace Test13 {

// Test that we don't try to emit a vtable for 'A' twice.
struct A {
  virtual void f();
};

struct B : virtual A {
  virtual void f();
};

// CHECK:      Vtable for 'Test13::C' (6 entries).
// CHECK-NEXT:    0 | vbase_offset (0)
// CHECK-NEXT:    1 | vbase_offset (0)
// CHECK-NEXT:    2 | vcall_offset (0)
// CHECK-NEXT:    3 | offset_to_top (0)
// CHECK-NEXT:    4 | Test13::C RTTI
// CHECK-NEXT:        -- (Test13::A, 0) vtable address --
// CHECK-NEXT:        -- (Test13::B, 0) vtable address --
// CHECK-NEXT:        -- (Test13::C, 0) vtable address --
// CHECK-NEXT:    5 | void Test13::C::f()
struct C : virtual B, virtual A {
  virtual void f();
};
void C::f() { }

}

namespace Test14 {

// Verify that we handle A being a non-virtual base of B, which is a virtual base.

struct A { 
  virtual void f(); 
};

struct B : A { };

struct C : virtual B { };

// CHECK:      Vtable for 'Test14::D' (5 entries).
// CHECK-NEXT:    0 | vbase_offset (0)
// CHECK-NEXT:    1 | vcall_offset (0)
// CHECK-NEXT:    2 | offset_to_top (0)
// CHECK-NEXT:    3 | Test14::D RTTI
// CHECK-NEXT:        -- (Test14::A, 0) vtable address --
// CHECK-NEXT:        -- (Test14::B, 0) vtable address --
// CHECK-NEXT:        -- (Test14::C, 0) vtable address --
// CHECK-NEXT:        -- (Test14::D, 0) vtable address --
// CHECK-NEXT:    4 | void Test14::D::f()
struct D : C, virtual B {
 virtual void f();
};
void D::f() { }

}

namespace Test15 {

// Test that we don't emit an extra vtable for B since it's a primary base of C.
struct A { virtual void a(); };
struct B { virtual void b(); };

struct C : virtual B { };

// CHECK:      Vtable for 'Test15::D' (11 entries).
// CHECK-NEXT:    0 | vbase_offset (8)
// CHECK-NEXT:    1 | vbase_offset (8)
// CHECK-NEXT:    2 | offset_to_top (0)
// CHECK-NEXT:    3 | Test15::D RTTI
// CHECK-NEXT:        -- (Test15::A, 0) vtable address --
// CHECK-NEXT:        -- (Test15::D, 0) vtable address --
// CHECK-NEXT:    4 | void Test15::A::a()
// CHECK-NEXT:    5 | void Test15::D::f()
// CHECK-NEXT:    6 | vbase_offset (0)
// CHECK-NEXT:    7 | vcall_offset (0)
// CHECK-NEXT:    8 | offset_to_top (-8)
// CHECK-NEXT:    9 | Test15::D RTTI
// CHECK-NEXT:        -- (Test15::B, 8) vtable address --
// CHECK-NEXT:        -- (Test15::C, 8) vtable address --
// CHECK-NEXT:   10 | void Test15::B::b()
struct D : A, virtual B, virtual C { 
  virtual void f();
};
void D::f() { } 

}

namespace Test16 {

// Test that destructors share vcall offsets.

struct A { virtual ~A(); };
struct B { virtual ~B(); };

struct C : A, B { virtual ~C(); };

// CHECK:      Vtable for 'Test16::D' (15 entries).
// CHECK-NEXT:    0 | vbase_offset (8)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test16::D RTTI
// CHECK-NEXT:        -- (Test16::D, 0) vtable address --
// CHECK-NEXT:    3 | void Test16::D::f()
// CHECK-NEXT:    4 | Test16::D::~D() [complete]
// CHECK-NEXT:    5 | Test16::D::~D() [deleting]
// CHECK-NEXT:    6 | vcall_offset (-8)
// CHECK-NEXT:    7 | offset_to_top (-8)
// CHECK-NEXT:    8 | Test16::D RTTI
// CHECK-NEXT:        -- (Test16::A, 8) vtable address --
// CHECK-NEXT:        -- (Test16::C, 8) vtable address --
// CHECK-NEXT:    9 | Test16::D::~D() [complete]
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-NEXT:   10 | Test16::D::~D() [deleting]
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-NEXT:   11 | offset_to_top (-16)
// CHECK-NEXT:   12 | Test16::D RTTI
// CHECK-NEXT:        -- (Test16::B, 16) vtable address --
// CHECK-NEXT:   13 | Test16::D::~D() [complete]
// CHECK-NEXT:        [this adjustment: -8 non-virtual, -24 vcall offset offset]
// CHECK-NEXT:   14 | Test16::D::~D() [deleting]
// CHECK-NEXT:        [this adjustment: -8 non-virtual, -24 vcall offset offset]
struct D : virtual C {
  virtual void f();
};
void D::f() { } 

}

namespace Test17 {

// Test that we don't mark E::f in the C-in-E vtable as unused.
struct A { virtual void f(); };
struct B : virtual A { virtual void f(); };
struct C : virtual A { virtual void f(); };
struct D : virtual B, virtual C { virtual void f(); };

// CHECK:      Vtable for 'Test17::E' (13 entries).
// CHECK-NEXT:    0 | vbase_offset (0)
// CHECK-NEXT:    1 | vbase_offset (8)
// CHECK-NEXT:    2 | vbase_offset (0)
// CHECK-NEXT:    3 | vbase_offset (0)
// CHECK-NEXT:    4 | vcall_offset (0)
// CHECK-NEXT:    5 | offset_to_top (0)
// CHECK-NEXT:    6 | Test17::E RTTI
// CHECK-NEXT:        -- (Test17::A, 0) vtable address --
// CHECK-NEXT:        -- (Test17::B, 0) vtable address --
// CHECK-NEXT:        -- (Test17::D, 0) vtable address --
// CHECK-NEXT:        -- (Test17::E, 0) vtable address --
// CHECK-NEXT:    7 | void Test17::E::f()
// CHECK-NEXT:    8 | vbase_offset (-8)
// CHECK-NEXT:    9 | vcall_offset (-8)
// CHECK-NEXT:   10 | offset_to_top (-8)
// CHECK-NEXT:   11 | Test17::E RTTI
// CHECK-NEXT:        -- (Test17::C, 8) vtable address --
// CHECK-NEXT:   12 | void Test17::E::f()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
class E : virtual D {
  virtual void f();  
};
void E::f() {}

}

namespace Test18 {

// Test that we compute the right 'this' adjustment offsets.

struct A {
  virtual void f();
  virtual void g();
};

struct B : virtual A {
  virtual void f();
};

struct C : A, B {
  virtual void g();
};

// CHECK:      Vtable for 'Test18::D' (24 entries).
// CHECK-NEXT:    0 | vbase_offset (8)
// CHECK-NEXT:    1 | vbase_offset (0)
// CHECK-NEXT:    2 | vbase_offset (0)
// CHECK-NEXT:    3 | vcall_offset (8)
// CHECK-NEXT:    4 | vcall_offset (0)
// CHECK-NEXT:    5 | offset_to_top (0)
// CHECK-NEXT:    6 | Test18::D RTTI
// CHECK-NEXT:        -- (Test18::A, 0) vtable address --
// CHECK-NEXT:        -- (Test18::B, 0) vtable address --
// CHECK-NEXT:        -- (Test18::D, 0) vtable address --
// CHECK-NEXT:    7 | void Test18::D::f()
// CHECK-NEXT:    8 | void Test18::C::g()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]
// CHECK-NEXT:    9 | void Test18::D::h()
// CHECK-NEXT:   10 | vcall_offset (0)
// CHECK-NEXT:   11 | vcall_offset (-8)
// CHECK-NEXT:   12 | vbase_offset (-8)
// CHECK-NEXT:   13 | offset_to_top (-8)
// CHECK-NEXT:   14 | Test18::D RTTI
// CHECK-NEXT:        -- (Test18::A, 8) vtable address --
// CHECK-NEXT:        -- (Test18::C, 8) vtable address --
// CHECK-NEXT:   15 | void Test18::D::f()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]
// CHECK-NEXT:   16 | void Test18::C::g()
// CHECK-NEXT:   17 | vbase_offset (-16)
// CHECK-NEXT:   18 | vcall_offset (-8)
// CHECK-NEXT:   19 | vcall_offset (-16)
// CHECK-NEXT:   20 | offset_to_top (-16)
// CHECK-NEXT:   21 | Test18::D RTTI
// CHECK-NEXT:        -- (Test18::B, 16) vtable address --
// CHECK-NEXT:   22 | void Test18::D::f()
// CHECK-NEXT:        [this adjustment: -8 non-virtual, -32 vcall offset offset]
// CHECK-NEXT:   23 | [unused] void Test18::C::g()

// CHECK:      Construction vtable for ('Test18::B', 0) in 'Test18::D' (7 entries).
// CHECK-NEXT:    0 | vbase_offset (0)
// CHECK-NEXT:    1 | vcall_offset (0)
// CHECK-NEXT:    2 | vcall_offset (0)
// CHECK-NEXT:    3 | offset_to_top (0)
// CHECK-NEXT:    4 | Test18::B RTTI
// CHECK-NEXT:        -- (Test18::A, 0) vtable address --
// CHECK-NEXT:        -- (Test18::B, 0) vtable address --
// CHECK-NEXT:    5 | void Test18::B::f()
// CHECK-NEXT:    6 | void Test18::A::g()

// CHECK:      Construction vtable for ('Test18::C', 8) in 'Test18::D' (20 entries).
// CHECK-NEXT:    0 | vcall_offset (0)
// CHECK-NEXT:    1 | vcall_offset (0)
// CHECK-NEXT:    2 | vbase_offset (-8)
// CHECK-NEXT:    3 | offset_to_top (0)
// CHECK-NEXT:    4 | Test18::C RTTI
// CHECK-NEXT:        -- (Test18::A, 8) vtable address --
// CHECK-NEXT:        -- (Test18::C, 8) vtable address --
// CHECK-NEXT:    5 | void Test18::A::f()
// CHECK-NEXT:    6 | void Test18::C::g()
// CHECK-NEXT:    7 | vbase_offset (-16)
// CHECK-NEXT:    8 | vcall_offset (-8)
// CHECK-NEXT:    9 | vcall_offset (0)
// CHECK-NEXT:   10 | offset_to_top (-8)
// CHECK-NEXT:   11 | Test18::C RTTI
// CHECK-NEXT:        -- (Test18::B, 16) vtable address --
// CHECK-NEXT:   12 | void Test18::B::f()
// CHECK-NEXT:   13 | [unused] void Test18::C::g()
// CHECK-NEXT:   14 | vcall_offset (8)
// CHECK-NEXT:   15 | vcall_offset (16)
// CHECK-NEXT:   16 | offset_to_top (8)
// CHECK-NEXT:   17 | Test18::C RTTI
// CHECK-NEXT:        -- (Test18::A, 0) vtable address --
// CHECK-NEXT:   18 | void Test18::B::f()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-NEXT:   19 | void Test18::C::g()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]

// CHECK:      Construction vtable for ('Test18::B', 16) in 'Test18::D' (13 entries).
// CHECK-NEXT:    0 | vbase_offset (-16)
// CHECK-NEXT:    1 | vcall_offset (-16)
// CHECK-NEXT:    2 | vcall_offset (0)
// CHECK-NEXT:    3 | offset_to_top (0)
// CHECK-NEXT:    4 | Test18::B RTTI
// CHECK-NEXT:        -- (Test18::B, 16) vtable address --
// CHECK-NEXT:    5 | void Test18::B::f()
// CHECK-NEXT:    6 | [unused] void Test18::A::g()
// CHECK-NEXT:    7 | vcall_offset (0)
// CHECK-NEXT:    8 | vcall_offset (16)
// CHECK-NEXT:    9 | offset_to_top (16)
// CHECK-NEXT:   10 | Test18::B RTTI
// CHECK-NEXT:        -- (Test18::A, 0) vtable address --
// CHECK-NEXT:   11 | void Test18::B::f()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-NEXT:   12 | void Test18::A::g()
struct D : virtual B, virtual C, virtual A 
{
  virtual void f();
  virtual void h();
};
void D::f() {}

}

namespace Test19 {

// Another 'this' adjustment test.

struct A {
  int a;

  virtual void f();
};

struct B : A {
  int b;

  virtual void g();
};

struct C {
  virtual void c();
};

// CHECK:      Vtable for 'Test19::D' (13 entries).
// CHECK-NEXT:    0 | vbase_offset (24)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test19::D RTTI
// CHECK-NEXT:        -- (Test19::C, 0) vtable address --
// CHECK-NEXT:        -- (Test19::D, 0) vtable address --
// CHECK-NEXT:    3 | void Test19::C::c()
// CHECK-NEXT:    4 | void Test19::D::f()
// CHECK-NEXT:    5 | offset_to_top (-8)
// CHECK-NEXT:    6 | Test19::D RTTI
// CHECK-NEXT:        -- (Test19::A, 8) vtable address --
// CHECK-NEXT:        -- (Test19::B, 8) vtable address --
// CHECK-NEXT:    7 | void Test19::D::f()
// CHECK-NEXT:        [this adjustment: -8 non-virtual]
// CHECK-NEXT:    8 | void Test19::B::g()
// CHECK-NEXT:    9 | vcall_offset (-24)
// CHECK-NEXT:   10 | offset_to_top (-24)
// CHECK-NEXT:   11 | Test19::D RTTI
// CHECK-NEXT:        -- (Test19::A, 24) vtable address --
// CHECK-NEXT:   12 | void Test19::D::f()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
struct D : C, B, virtual A {
  virtual void f();
};
void D::f() { }

}

namespace Test20 {

// pure virtual member functions should never have 'this' adjustments.

struct A {
  virtual void f() = 0;
  virtual void g();
};

struct B : A { };

// CHECK:      Vtable for 'Test20::C' (9 entries).
// CHECK-NEXT:    0 | offset_to_top (0)
// CHECK-NEXT:    1 | Test20::C RTTI
// CHECK-NEXT:        -- (Test20::A, 0) vtable address --
// CHECK-NEXT:        -- (Test20::C, 0) vtable address --
// CHECK-NEXT:    2 | void Test20::C::f() [pure]
// CHECK-NEXT:    3 | void Test20::A::g()
// CHECK-NEXT:    4 | void Test20::C::h()
// CHECK-NEXT:    5 | offset_to_top (-8)
// CHECK-NEXT:    6 | Test20::C RTTI
// CHECK-NEXT:        -- (Test20::A, 8) vtable address --
// CHECK-NEXT:        -- (Test20::B, 8) vtable address --
// CHECK-NEXT:    7 | void Test20::C::f() [pure]
// CHECK-NEXT:    8 | void Test20::A::g()
struct C : A, B { 
  virtual void f() = 0;
  virtual void h();
};
void C::h() { }

}

namespace Test21 {

// Test that we get vbase offsets right in secondary vtables.
struct A { 
  virtual void f();
};

struct B : virtual A { };
class C : virtual B { };
class D : virtual C { };

class E : virtual C { };

// CHECK:      Vtable for 'Test21::F' (16 entries).
// CHECK-NEXT:    0 | vbase_offset (8)
// CHECK-NEXT:    1 | vbase_offset (0)
// CHECK-NEXT:    2 | vbase_offset (0)
// CHECK-NEXT:    3 | vbase_offset (0)
// CHECK-NEXT:    4 | vbase_offset (0)
// CHECK-NEXT:    5 | vcall_offset (0)
// CHECK-NEXT:    6 | offset_to_top (0)
// CHECK-NEXT:    7 | Test21::F RTTI
// CHECK-NEXT:        -- (Test21::A, 0) vtable address --
// CHECK-NEXT:        -- (Test21::B, 0) vtable address --
// CHECK-NEXT:        -- (Test21::C, 0) vtable address --
// CHECK-NEXT:        -- (Test21::D, 0) vtable address --
// CHECK-NEXT:        -- (Test21::F, 0) vtable address --
// CHECK-NEXT:    8 | void Test21::F::f()
// CHECK-NEXT:    9 | vbase_offset (-8)
// CHECK-NEXT:   10 | vbase_offset (-8)
// CHECK-NEXT:   11 | vbase_offset (-8)
// CHECK-NEXT:   12 | vcall_offset (-8)
// CHECK-NEXT:   13 | offset_to_top (-8)
// CHECK-NEXT:   14 | Test21::F RTTI
// CHECK-NEXT:        -- (Test21::E, 8) vtable address --
// CHECK-NEXT:   15 | [unused] void Test21::F::f()
//
// CHECK:      Virtual base offset offsets for 'Test21::F' (5 entries).
// CHECK-NEXT:    Test21::A | -32
// CHECK-NEXT:    Test21::B | -40
// CHECK-NEXT:    Test21::C | -48
// CHECK-NEXT:    Test21::D | -56
// CHECK-NEXT:    Test21::E | -64
class F : virtual D, virtual E {
  virtual void f();
};
void F::f() { }

}

namespace Test22 {

// Very simple construction vtable test.
struct V1 {
  int v1;
}; 

struct V2 : virtual V1 {
  int v2; 
};

// CHECK:      Vtable for 'Test22::C' (8 entries).
// CHECK-NEXT:    0 | vbase_offset (16)
// CHECK-NEXT:    1 | vbase_offset (12)
// CHECK-NEXT:    2 | offset_to_top (0)
// CHECK-NEXT:    3 | Test22::C RTTI
// CHECK-NEXT:        -- (Test22::C, 0) vtable address --
// CHECK-NEXT:    4 | void Test22::C::f()
// CHECK-NEXT:    5 | vbase_offset (-4)
// CHECK-NEXT:    6 | offset_to_top (-16)
// CHECK-NEXT:    7 | Test22::C RTTI
// CHECK-NEXT:        -- (Test22::V2, 16) vtable address --

// CHECK:      Construction vtable for ('Test22::V2', 16) in 'Test22::C' (3 entries).
// CHECK-NEXT:    0 | vbase_offset (-4)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test22::V2 RTTI

struct C : virtual V1, virtual V2 {
  int c; 
  virtual void f(); 
};
void C::f() { } 

}

namespace Test23 {

struct A {
  int a;
};

struct B : virtual A {
  int b;
};

struct C : A, virtual B {
  int c;
};

// CHECK:      Vtable for 'Test23::D' (7 entries).
// CHECK-NEXT:    0 | vbase_offset (20)
// CHECK-NEXT:    1 | vbase_offset (24)
// CHECK-NEXT:    2 | offset_to_top (0)
// CHECK-NEXT:    3 | Test23::D RTTI
// CHECK-NEXT:        -- (Test23::C, 0) vtable address --
// CHECK-NEXT:        -- (Test23::D, 0) vtable address --
// CHECK-NEXT:    4 | vbase_offset (-4)
// CHECK-NEXT:    5 | offset_to_top (-24)
// CHECK-NEXT:    6 | Test23::D RTTI
// CHECK-NEXT:        -- (Test23::B, 24) vtable address --

// CHECK:      Construction vtable for ('Test23::C', 0) in 'Test23::D' (7 entries).
// CHECK-NEXT:    0 | vbase_offset (20)
// CHECK-NEXT:    1 | vbase_offset (24)
// CHECK-NEXT:    2 | offset_to_top (0)
// CHECK-NEXT:    3 | Test23::C RTTI
// CHECK-NEXT:        -- (Test23::C, 0) vtable address --
// CHECK-NEXT:    4 | vbase_offset (-4)
// CHECK-NEXT:    5 | offset_to_top (-24)
// CHECK-NEXT:    6 | Test23::C RTTI
// CHECK-NEXT:        -- (Test23::B, 24) vtable address --

// CHECK:      Construction vtable for ('Test23::B', 24) in 'Test23::D' (3 entries).
// CHECK-NEXT:    0 | vbase_offset (-4)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test23::B RTTI
// CHECK-NEXT:        -- (Test23::B, 24) vtable address --

struct D : virtual A, virtual B, C {
  int d;

  void f();
};
void D::f() { } 

}

namespace Test24 {

// Another construction vtable test.

struct A {
  virtual void f();
};

struct B : virtual A { };
struct C : virtual A { };

// CHECK:      Vtable for 'Test24::D' (10 entries).
// CHECK-NEXT:    0 | vbase_offset (0)
// CHECK-NEXT:    1 | vcall_offset (0)
// CHECK-NEXT:    2 | offset_to_top (0)
// CHECK-NEXT:    3 | Test24::D RTTI
// CHECK-NEXT:        -- (Test24::A, 0) vtable address --
// CHECK-NEXT:        -- (Test24::B, 0) vtable address --
// CHECK-NEXT:        -- (Test24::D, 0) vtable address --
// CHECK-NEXT:    4 | void Test24::D::f()
// CHECK-NEXT:    5 | vbase_offset (-8)
// CHECK-NEXT:    6 | vcall_offset (-8)
// CHECK-NEXT:    7 | offset_to_top (-8)
// CHECK-NEXT:    8 | Test24::D RTTI
// CHECK-NEXT:        -- (Test24::C, 8) vtable address --
// CHECK-NEXT:    9 | [unused] void Test24::D::f()

// CHECK:      Construction vtable for ('Test24::B', 0) in 'Test24::D' (5 entries).
// CHECK-NEXT:    0 | vbase_offset (0)
// CHECK-NEXT:    1 | vcall_offset (0)
// CHECK-NEXT:    2 | offset_to_top (0)
// CHECK-NEXT:    3 | Test24::B RTTI
// CHECK-NEXT:        -- (Test24::A, 0) vtable address --
// CHECK-NEXT:        -- (Test24::B, 0) vtable address --
// CHECK-NEXT:    4 | void Test24::A::f()

// CHECK:      Construction vtable for ('Test24::C', 8) in 'Test24::D' (9 entries).
// CHECK-NEXT:    0 | vbase_offset (-8)
// CHECK-NEXT:    1 | vcall_offset (-8)
// CHECK-NEXT:    2 | offset_to_top (0)
// CHECK-NEXT:    3 | Test24::C RTTI
// CHECK-NEXT:        -- (Test24::C, 8) vtable address --
// CHECK-NEXT:    4 | [unused] void Test24::A::f()
// CHECK-NEXT:    5 | vcall_offset (0)
// CHECK-NEXT:    6 | offset_to_top (8)
// CHECK-NEXT:    7 | Test24::C RTTI
// CHECK-NEXT:        -- (Test24::A, 0) vtable address --
// CHECK-NEXT:    8 | void Test24::A::f()
struct D : B, C {
  virtual void f();
};
void D::f() { }

}

namespace Test25 {
  
// This mainly tests that we don't assert on this class hierarchy.

struct V {
  virtual void f();
};

struct A : virtual V { };
struct B : virtual V { };

// CHECK:      Vtable for 'Test25::C' (11 entries).
// CHECK-NEXT:    0 | vbase_offset (0)
// CHECK-NEXT:    1 | vcall_offset (0)
// CHECK-NEXT:    2 | offset_to_top (0)
// CHECK-NEXT:    3 | Test25::C RTTI
// CHECK-NEXT:        -- (Test25::A, 0) vtable address --
// CHECK-NEXT:        -- (Test25::C, 0) vtable address --
// CHECK-NEXT:        -- (Test25::V, 0) vtable address --
// CHECK-NEXT:    4 | void Test25::V::f()
// CHECK-NEXT:    5 | void Test25::C::g()
// CHECK-NEXT:    6 | vbase_offset (-8)
// CHECK-NEXT:    7 | vcall_offset (-8)
// CHECK-NEXT:    8 | offset_to_top (-8)
// CHECK-NEXT:    9 | Test25::C RTTI
// CHECK-NEXT:        -- (Test25::B, 8) vtable address --
// CHECK-NEXT:   10 | [unused] void Test25::V::f()

// CHECK:      Construction vtable for ('Test25::A', 0) in 'Test25::C' (5 entries).
// CHECK-NEXT:    0 | vbase_offset (0)
// CHECK-NEXT:    1 | vcall_offset (0)
// CHECK-NEXT:    2 | offset_to_top (0)
// CHECK-NEXT:    3 | Test25::A RTTI
// CHECK-NEXT:        -- (Test25::A, 0) vtable address --
// CHECK-NEXT:        -- (Test25::V, 0) vtable address --
// CHECK-NEXT:    4 | void Test25::V::f()

// CHECK:      Construction vtable for ('Test25::B', 8) in 'Test25::C' (9 entries).
// CHECK-NEXT:    0 | vbase_offset (-8)
// CHECK-NEXT:    1 | vcall_offset (-8)
// CHECK-NEXT:    2 | offset_to_top (0)
// CHECK-NEXT:    3 | Test25::B RTTI
// CHECK-NEXT:        -- (Test25::B, 8) vtable address --
// CHECK-NEXT:    4 | [unused] void Test25::V::f()
// CHECK-NEXT:    5 | vcall_offset (0)
// CHECK-NEXT:    6 | offset_to_top (8)
// CHECK-NEXT:    7 | Test25::B RTTI
// CHECK-NEXT:        -- (Test25::V, 0) vtable address --
// CHECK-NEXT:    8 | void Test25::V::f()
struct C : A, virtual V, B {
  virtual void g();
};
void C::g() { }

}

namespace Test26 {

// Test that we generate the right number of entries in the C-in-D construction vtable, and that
// we don't mark A::a as unused.

struct A {
  virtual void a();
};

struct B {
  virtual void c();
};

struct C : virtual A {
  virtual void b();
};

// CHECK:      Vtable for 'Test26::D' (15 entries).
// CHECK-NEXT:    0 | vbase_offset (8)
// CHECK-NEXT:    1 | vbase_offset (8)
// CHECK-NEXT:    2 | vbase_offset (0)
// CHECK-NEXT:    3 | vcall_offset (0)
// CHECK-NEXT:    4 | offset_to_top (0)
// CHECK-NEXT:    5 | Test26::D RTTI
// CHECK-NEXT:        -- (Test26::B, 0) vtable address --
// CHECK-NEXT:        -- (Test26::D, 0) vtable address --
// CHECK-NEXT:    6 | void Test26::B::c()
// CHECK-NEXT:    7 | void Test26::D::d()
// CHECK-NEXT:    8 | vcall_offset (0)
// CHECK-NEXT:    9 | vbase_offset (0)
// CHECK-NEXT:   10 | vcall_offset (0)
// CHECK-NEXT:   11 | offset_to_top (-8)
// CHECK-NEXT:   12 | Test26::D RTTI
// CHECK-NEXT:        -- (Test26::A, 8) vtable address --
// CHECK-NEXT:        -- (Test26::C, 8) vtable address --
// CHECK-NEXT:   13 | void Test26::A::a()
// CHECK-NEXT:   14 | void Test26::C::b()

// CHECK:      Construction vtable for ('Test26::C', 8) in 'Test26::D' (7 entries).
// CHECK-NEXT:    0 | vcall_offset (0)
// CHECK-NEXT:    1 | vbase_offset (0)
// CHECK-NEXT:    2 | vcall_offset (0)
// CHECK-NEXT:    3 | offset_to_top (0)
// CHECK-NEXT:    4 | Test26::C RTTI
// CHECK-NEXT:        -- (Test26::A, 8) vtable address --
// CHECK-NEXT:        -- (Test26::C, 8) vtable address --
// CHECK-NEXT:    5 | void Test26::A::a()
// CHECK-NEXT:    6 | void Test26::C::b()
class D : virtual B, virtual C {
  virtual void d();
};
void D::d() { } 

}

namespace Test27 {

// Test that we don't generate a secondary vtable for C in the D-in-E vtable, since
// C doesn't have any virtual bases.

struct A {
  virtual void a();
};

struct B {
  virtual void b();
};

struct C {
  virtual void c();
};

struct D : A, virtual B, C {
  virtual void d();
};

// CHECK:      Vtable for 'Test27::E' (13 entries).
// CHECK-NEXT:    0 | vbase_offset (16)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test27::E RTTI
// CHECK-NEXT:        -- (Test27::A, 0) vtable address --
// CHECK-NEXT:        -- (Test27::D, 0) vtable address --
// CHECK-NEXT:        -- (Test27::E, 0) vtable address --
// CHECK-NEXT:    3 | void Test27::A::a()
// CHECK-NEXT:    4 | void Test27::D::d()
// CHECK-NEXT:    5 | void Test27::E::e()
// CHECK-NEXT:    6 | offset_to_top (-8)
// CHECK-NEXT:    7 | Test27::E RTTI
// CHECK-NEXT:        -- (Test27::C, 8) vtable address --
// CHECK-NEXT:    8 | void Test27::C::c()
// CHECK-NEXT:    9 | vcall_offset (0)
// CHECK-NEXT:   10 | offset_to_top (-16)
// CHECK-NEXT:   11 | Test27::E RTTI
// CHECK-NEXT:        -- (Test27::B, 16) vtable address --
// CHECK-NEXT:   12 | void Test27::B::b()

// CHECK:      Construction vtable for ('Test27::D', 0) in 'Test27::E' (9 entries).
// CHECK-NEXT:    0 | vbase_offset (16)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test27::D RTTI
// CHECK-NEXT:        -- (Test27::A, 0) vtable address --
// CHECK-NEXT:        -- (Test27::D, 0) vtable address --
// CHECK-NEXT:    3 | void Test27::A::a()
// CHECK-NEXT:    4 | void Test27::D::d()
// CHECK-NEXT:    5 | vcall_offset (0)
// CHECK-NEXT:    6 | offset_to_top (-16)
// CHECK-NEXT:    7 | Test27::D RTTI
// CHECK-NEXT:        -- (Test27::B, 16) vtable address --
// CHECK-NEXT:    8 | void Test27::B::b()
struct E : D {
  virtual void e();
};
void E::e() { }

}

namespace Test28 {

// Check that we do include the vtable for B in the D-in-E construction vtable, since
// B is a base class of a virtual base (C).

struct A {
  virtual void a();
};

struct B {
  virtual void b();
};

struct C : A, B {
  virtual void c();
};

struct D : virtual C {
};

// CHECK:      Vtable for 'Test28::E' (14 entries).
// CHECK-NEXT:    0 | vbase_offset (8)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test28::E RTTI
// CHECK-NEXT:        -- (Test28::D, 0) vtable address --
// CHECK-NEXT:        -- (Test28::E, 0) vtable address --
// CHECK-NEXT:    3 | void Test28::E::e()
// CHECK-NEXT:    4 | vcall_offset (8)
// CHECK-NEXT:    5 | vcall_offset (0)
// CHECK-NEXT:    6 | vcall_offset (0)
// CHECK-NEXT:    7 | offset_to_top (-8)
// CHECK-NEXT:    8 | Test28::E RTTI
// CHECK-NEXT:        -- (Test28::A, 8) vtable address --
// CHECK-NEXT:        -- (Test28::C, 8) vtable address --
// CHECK-NEXT:    9 | void Test28::A::a()
// CHECK-NEXT:   10 | void Test28::C::c()
// CHECK-NEXT:   11 | offset_to_top (-16)
// CHECK-NEXT:   12 | Test28::E RTTI
// CHECK-NEXT:        -- (Test28::B, 16) vtable address --
// CHECK-NEXT:   13 | void Test28::B::b()

// CHECK:      Construction vtable for ('Test28::D', 0) in 'Test28::E' (13 entries).
// CHECK-NEXT:    0 | vbase_offset (8)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test28::D RTTI
// CHECK-NEXT:        -- (Test28::D, 0) vtable address --
// CHECK-NEXT:    3 | vcall_offset (8)
// CHECK-NEXT:    4 | vcall_offset (0)
// CHECK-NEXT:    5 | vcall_offset (0)
// CHECK-NEXT:    6 | offset_to_top (-8)
// CHECK-NEXT:    7 | Test28::D RTTI
// CHECK-NEXT:        -- (Test28::A, 8) vtable address --
// CHECK-NEXT:        -- (Test28::C, 8) vtable address --
// CHECK-NEXT:    8 | void Test28::A::a()
// CHECK-NEXT:    9 | void Test28::C::c()
// CHECK-NEXT:   10 | offset_to_top (-16)
// CHECK-NEXT:   11 | Test28::D RTTI
// CHECK-NEXT:        -- (Test28::B, 16) vtable address --
// CHECK-NEXT:   12 | void Test28::B::b()
struct E : D {
  virtual void e();
};
void E::e() { }

}

namespace Test29 {

// Test that the covariant return thunk for B::f will have a virtual 'this' adjustment,
// matching gcc.

struct V1 { };
struct V2 : virtual V1 { };

struct A {
  virtual V1 *f();
};

// CHECK:      Vtable for 'Test29::B' (6 entries).
// CHECK-NEXT:    0 | vbase_offset (0)
// CHECK-NEXT:    1 | vcall_offset (0)
// CHECK-NEXT:    2 | offset_to_top (0)
// CHECK-NEXT:    3 | Test29::B RTTI
// CHECK-NEXT:        -- (Test29::A, 0) vtable address --
// CHECK-NEXT:        -- (Test29::B, 0) vtable address --
// CHECK-NEXT:    4 | Test29::V2 *Test29::B::f()
// CHECK-NEXT:        [return adjustment: 0 non-virtual, -24 vbase offset offset]
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-NEXT:    5 | Test29::V2 *Test29::B::f()
struct B : virtual A {
  virtual V2 *f();
};
V2 *B::f() { return 0; }

}

namespace Test30 {

// Test that we don't assert when generating a vtable for F.
struct A { };

struct B : virtual A {
 int i;
};

struct C {
 virtual void f();
};

struct D : virtual C, B { };
struct E : virtual D { };

struct F : E {
 virtual void f();
};
void F::f() { }

}

namespace Test31 {

// Test that we don't add D::f twice to the primary vtable.
struct A {
  int a;
};

struct B {
  virtual void f();
};

struct C : A, virtual B {
  virtual void f();
};

// CHECK:      Vtable for 'Test31::D' (11 entries).
// CHECK-NEXT:    0 | vbase_offset (0)
// CHECK-NEXT:    1 | vbase_offset (8)
// CHECK-NEXT:    2 | vcall_offset (0)
// CHECK-NEXT:    3 | offset_to_top (0)
// CHECK-NEXT:    4 | Test31::D RTTI
// CHECK-NEXT:        -- (Test31::B, 0) vtable address --
// CHECK-NEXT:        -- (Test31::D, 0) vtable address --
// CHECK-NEXT:    5 | void Test31::D::f()
// CHECK-NEXT:    6 | vbase_offset (-8)
// CHECK-NEXT:    7 | vcall_offset (-8)
// CHECK-NEXT:    8 | offset_to_top (-8)
// CHECK-NEXT:    9 | Test31::D RTTI
// CHECK-NEXT:        -- (Test31::C, 8) vtable address --
// CHECK-NEXT:   10 | void Test31::D::f()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
struct D : virtual C {
  virtual void f();
};
void D::f() { }

}

namespace Test32 {

// Check that we correctly lay out the virtual bases of 'Test32::D'.

struct A {
  virtual void f();
};

struct B : virtual A { };
struct C : A, virtual B { };
struct D : virtual B { };

// CHECK:      Virtual base offset offsets for 'Test32::E' (3 entries).
// CHECK-NEXT:    Test32::A | -32
// CHECK-NEXT:    Test32::B | -24
// CHECK-NEXT:    Test32::D | -40
struct E : C, virtual D {
  virtual void f();
};
void E::f() { }

}

namespace Test33 {

// Test that we don't emit too many vcall offsets in 'Test32::F'.

struct A {
  virtual void a();
};

struct B {
  virtual void b();
};

struct C : virtual A, virtual B {
  virtual void c();
};

struct D : virtual C { };

struct E : A, D { 
  virtual void e();
};

// CHECK:      Vtable for 'Test33::F' (30 entries).
// CHECK-NEXT:    0 | vbase_offset (24)
// CHECK-NEXT:    1 | vbase_offset (16)
// CHECK-NEXT:    2 | vbase_offset (16)
// CHECK-NEXT:    3 | vbase_offset (8)
// CHECK-NEXT:    4 | offset_to_top (0)
// CHECK-NEXT:    5 | Test33::F RTTI
// CHECK-NEXT:        -- (Test33::A, 0) vtable address --
// CHECK-NEXT:        -- (Test33::F, 0) vtable address --
// CHECK-NEXT:    6 | void Test33::A::a()
// CHECK-NEXT:    7 | void Test33::F::f()
// CHECK-NEXT:    8 | vcall_offset (0)
// CHECK-NEXT:    9 | vcall_offset (0)
// CHECK-NEXT:   10 | vbase_offset (16)
// CHECK-NEXT:   11 | vbase_offset (8)
// CHECK-NEXT:   12 | vbase_offset (8)
// CHECK-NEXT:   13 | offset_to_top (-8)
// CHECK-NEXT:   14 | Test33::F RTTI
// CHECK-NEXT:        -- (Test33::A, 8) vtable address --
// CHECK-NEXT:        -- (Test33::E, 8) vtable address --
// CHECK-NEXT:   15 | void Test33::A::a()
// CHECK-NEXT:   16 | void Test33::E::e()
// CHECK-NEXT:   17 | vbase_offset (0)
// CHECK-NEXT:   18 | vcall_offset (0)
// CHECK-NEXT:   19 | vbase_offset (8)
// CHECK-NEXT:   20 | vbase_offset (0)
// CHECK-NEXT:   21 | vcall_offset (0)
// CHECK-NEXT:   22 | offset_to_top (-16)
// CHECK-NEXT:   23 | Test33::F RTTI
// CHECK-NEXT:        -- (Test33::A, 16) vtable address --
// CHECK-NEXT:        -- (Test33::C, 16) vtable address --
// CHECK-NEXT:        -- (Test33::D, 16) vtable address --
// CHECK-NEXT:   24 | void Test33::A::a()
// CHECK-NEXT:   25 | void Test33::C::c()
// CHECK-NEXT:   26 | vcall_offset (0)
// CHECK-NEXT:   27 | offset_to_top (-24)
// CHECK-NEXT:   28 | Test33::F RTTI
// CHECK-NEXT:        -- (Test33::B, 24) vtable address --
// CHECK-NEXT:   29 | void Test33::B::b()
struct F : virtual E, A {
  virtual void f();
};
void F::f() { }

}

namespace Test34 {

// Test that we lay out the construction vtable for 'Test34::E' in 'Test34::F' correctly.

struct A {
  virtual void a();
};
struct B : virtual A { };

struct C : B, A {
  virtual void c();
};

struct D : A, C { };

struct E : virtual D {
  virtual void e();
};

// CHECK:      Construction vtable for ('Test34::E', 0) in 'Test34::F' (22 entries).
// CHECK-NEXT:    0 | vbase_offset (0)
// CHECK-NEXT:    1 | vbase_offset (8)
// CHECK-NEXT:    2 | vcall_offset (0)
// CHECK-NEXT:    3 | offset_to_top (0)
// CHECK-NEXT:    4 | Test34::E RTTI
// CHECK-NEXT:        -- (Test34::A, 0) vtable address --
// CHECK-NEXT:        -- (Test34::E, 0) vtable address --
// CHECK-NEXT:    5 | void Test34::A::a()
// CHECK-NEXT:    6 | void Test34::E::e()
// CHECK-NEXT:    7 | vcall_offset (8)
// CHECK-NEXT:    8 | vcall_offset (0)
// CHECK-NEXT:    9 | vbase_offset (-8)
// CHECK-NEXT:   10 | offset_to_top (-8)
// CHECK-NEXT:   11 | Test34::E RTTI
// CHECK-NEXT:        -- (Test34::A, 8) vtable address --
// CHECK-NEXT:        -- (Test34::D, 8) vtable address --
// CHECK-NEXT:   12 | void Test34::A::a()
// CHECK-NEXT:   13 | vbase_offset (-16)
// CHECK-NEXT:   14 | vcall_offset (-16)
// CHECK-NEXT:   15 | offset_to_top (-16)
// CHECK-NEXT:   16 | Test34::E RTTI
// CHECK-NEXT:        -- (Test34::B, 16) vtable address --
// CHECK-NEXT:        -- (Test34::C, 16) vtable address --
// CHECK-NEXT:   17 | [unused] void Test34::A::a()
// CHECK-NEXT:   18 | void Test34::C::c()
// CHECK-NEXT:   19 | offset_to_top (-24)
// CHECK-NEXT:   20 | Test34::E RTTI
// CHECK-NEXT:        -- (Test34::A, 24) vtable address --
// CHECK-NEXT:   21 | void Test34::A::a()
struct F : E {
  virtual void f();
};
void F::f() { }

}

namespace Test35 {

// Test that we lay out the virtual bases of 'Test35::H' in the correct order.

struct A {
 virtual void a();

 int i;
};

struct B : virtual A {
 virtual void b();
};

struct C {
 virtual void c();
};

struct D : C, virtual B {
 virtual void d();
};

struct E : D {
 virtual void e();

 bool b;
};

struct F : virtual D { };
struct G : virtual E { };

// CHECK:      Vtable for 'Test35::H' (32 entries).
// CHECK-NEXT:    0 | vbase_offset (32)
// CHECK-NEXT:    1 | vbase_offset (0)
// CHECK-NEXT:    2 | vcall_offset (0)
// CHECK-NEXT:    3 | vcall_offset (0)
// CHECK-NEXT:    4 | vbase_offset (16)
// CHECK-NEXT:    5 | vbase_offset (8)
// CHECK-NEXT:    6 | offset_to_top (0)
// CHECK-NEXT:    7 | Test35::H RTTI
// CHECK-NEXT:        -- (Test35::C, 0) vtable address --
// CHECK-NEXT:        -- (Test35::D, 0) vtable address --
// CHECK-NEXT:        -- (Test35::F, 0) vtable address --
// CHECK-NEXT:        -- (Test35::H, 0) vtable address --
// CHECK-NEXT:    8 | void Test35::C::c()
// CHECK-NEXT:    9 | void Test35::D::d()
// CHECK-NEXT:   10 | void Test35::H::h()
// CHECK-NEXT:   11 | vbase_offset (0)
// CHECK-NEXT:   12 | vbase_offset (24)
// CHECK-NEXT:   13 | vcall_offset (0)
// CHECK-NEXT:   14 | vbase_offset (8)
// CHECK-NEXT:   15 | offset_to_top (-8)
// CHECK-NEXT:   16 | Test35::H RTTI
// CHECK-NEXT:        -- (Test35::B, 8) vtable address --
// CHECK-NEXT:        -- (Test35::G, 8) vtable address --
// CHECK-NEXT:   17 | void Test35::B::b()
// CHECK-NEXT:   18 | vcall_offset (0)
// CHECK-NEXT:   19 | offset_to_top (-16)
// CHECK-NEXT:   20 | Test35::H RTTI
// CHECK-NEXT:        -- (Test35::A, 16) vtable address --
// CHECK-NEXT:   21 | void Test35::A::a()
// CHECK-NEXT:   22 | vcall_offset (0)
// CHECK-NEXT:   23 | vcall_offset (0)
// CHECK-NEXT:   24 | vcall_offset (0)
// CHECK-NEXT:   25 | vbase_offset (-16)
// CHECK-NEXT:   26 | vbase_offset (-24)
// CHECK-NEXT:   27 | offset_to_top (-32)
// CHECK-NEXT:   28 | Test35::H RTTI
// CHECK-NEXT:        -- (Test35::C, 32) vtable address --
// CHECK-NEXT:        -- (Test35::D, 32) vtable address --
// CHECK-NEXT:        -- (Test35::E, 32) vtable address --
// CHECK-NEXT:   29 | void Test35::C::c()
// CHECK-NEXT:   30 | void Test35::D::d()
// CHECK-NEXT:   31 | void Test35::E::e()

// CHECK:      Virtual base offset offsets for 'Test35::H' (4 entries).
// CHECK-NEXT:    Test35::A | -32
// CHECK-NEXT:    Test35::B | -24
// CHECK-NEXT:    Test35::D | -56
// CHECK-NEXT:    Test35::E | -64
struct H : F, G {
 virtual void h();
};
void H::h() { }

}

namespace Test36 {

// Test that we don't mark B::f as unused in the vtable for D.

struct A {
 virtual void f();
};

struct B : virtual A { };

struct C : virtual A {
 virtual void f();
};

// CHECK:      Vtable for 'Test36::D' (12 entries).
// CHECK-NEXT:    0 | vbase_offset (8)
// CHECK-NEXT:    1 | vbase_offset (8)
// CHECK-NEXT:    2 | vcall_offset (0)
// CHECK-NEXT:    3 | offset_to_top (0)
// CHECK-NEXT:    4 | Test36::D RTTI
// CHECK-NEXT:        -- (Test36::C, 0) vtable address --
// CHECK-NEXT:        -- (Test36::D, 0) vtable address --
// CHECK-NEXT:    5 | void Test36::C::f()
// CHECK-NEXT:    6 | void Test36::D::g()
// CHECK-NEXT:    7 | vbase_offset (0)
// CHECK-NEXT:    8 | vcall_offset (-8)
// CHECK-NEXT:    9 | offset_to_top (-8)
// CHECK-NEXT:   10 | Test36::D RTTI
// CHECK-NEXT:        -- (Test36::A, 8) vtable address --
// CHECK-NEXT:        -- (Test36::B, 8) vtable address --
// CHECK-NEXT:   11 | void Test36::C::f()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
struct D : virtual B, C {
 virtual void g();
};
void D::g() { }

}
