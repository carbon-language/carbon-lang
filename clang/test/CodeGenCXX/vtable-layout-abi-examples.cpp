// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm-only -fdump-vtable-layouts 2>&1 | FileCheck %s

/// Examples from the Itanium C++ ABI specification.
/// http://www.codesourcery.com/public/cxx-abi/

namespace Test1 {
  
// This is from http://www.codesourcery.com/public/cxx-abi/cxx-vtable-ex.html

// CHECK:      Vtable for 'Test1::A' (5 entries).
// CHECK-NEXT:    0 | offset_to_top (0)
// CHECK-NEXT:    1 | Test1::A RTTI
// CHECK-NEXT:        -- (Test1::A, 0) vtable address --
// CHECK-NEXT:    2 | void Test1::A::f()
// CHECK-NEXT:    3 | void Test1::A::g()
// CHECK-NEXT:    4 | void Test1::A::h()
struct A {
  virtual void f ();
  virtual void g ();
  virtual void h ();
  int ia;
};
void A::f() {}

// CHECK:      Vtable for 'Test1::B' (13 entries).
// CHECK-NEXT:    0 | vbase_offset (16)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test1::B RTTI
// CHECK-NEXT:        -- (Test1::B, 0) vtable address --
// CHECK-NEXT:    3 | void Test1::B::f()
// CHECK-NEXT:    4 | void Test1::B::h()
// CHECK-NEXT:    5 | vcall_offset (-16)
// CHECK-NEXT:    6 | vcall_offset (0)
// CHECK-NEXT:    7 | vcall_offset (-16)
// CHECK-NEXT:    8 | offset_to_top (-16)
// CHECK-NEXT:    9 | Test1::B RTTI
// CHECK-NEXT:        -- (Test1::A, 16) vtable address --
// CHECK-NEXT:   10 | void Test1::B::f()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-NEXT:   11 | void Test1::A::g()
// CHECK-NEXT:   12 | void Test1::B::h()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -40 vcall offset offset]
struct B: public virtual A {
  void f ();
  void h ();
  int ib;
};
void B::f() {}

// CHECK:      Vtable for 'Test1::C' (13 entries).
// CHECK-NEXT:    0 | vbase_offset (16)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test1::C RTTI
// CHECK-NEXT:        -- (Test1::C, 0) vtable address --
// CHECK-NEXT:    3 | void Test1::C::g()
// CHECK-NEXT:    4 | void Test1::C::h()
// CHECK-NEXT:    5 | vcall_offset (-16)
// CHECK-NEXT:    6 | vcall_offset (-16)
// CHECK-NEXT:    7 | vcall_offset (0)
// CHECK-NEXT:    8 | offset_to_top (-16)
// CHECK-NEXT:    9 | Test1::C RTTI
// CHECK-NEXT:        -- (Test1::A, 16) vtable address --
// CHECK-NEXT:   10 | void Test1::A::f()
// CHECK-NEXT:   11 | void Test1::C::g()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]
// CHECK-NEXT:   12 | void Test1::C::h()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -40 vcall offset offset]
struct C: public virtual A {
  void g ();
  void h ();
  int ic;
};
void C::g() {}

// CHECK:      Vtable for 'Test1::D' (18 entries).
// CHECK-NEXT:    0 | vbase_offset (32)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test1::D RTTI
// CHECK-NEXT:        -- (Test1::B, 0) vtable address --
// CHECK-NEXT:        -- (Test1::D, 0) vtable address --
// CHECK-NEXT:    3 | void Test1::B::f()
// CHECK-NEXT:    4 | void Test1::D::h()
// CHECK-NEXT:    5 | vbase_offset (16)
// CHECK-NEXT:    6 | offset_to_top (-16)
// CHECK-NEXT:    7 | Test1::D RTTI
// CHECK-NEXT:        -- (Test1::C, 16) vtable address --
// CHECK-NEXT:    8 | void Test1::C::g()
// CHECK-NEXT:    9 | void Test1::D::h()
// CHECK-NEXT:        [this adjustment: -16 non-virtual]
// CHECK-NEXT:   10 | vcall_offset (-32)
// CHECK-NEXT:   11 | vcall_offset (-16)
// CHECK-NEXT:   12 | vcall_offset (-32)
// CHECK-NEXT:   13 | offset_to_top (-32)
// CHECK-NEXT:   14 | Test1::D RTTI
// CHECK-NEXT:        -- (Test1::A, 32) vtable address --
// CHECK-NEXT:   15 | void Test1::B::f()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-NEXT:   16 | void Test1::C::g()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]
// CHECK-NEXT:   17 | void Test1::D::h()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -40 vcall offset offset]
struct D: public B, public C {
  void h ();
  int id;
};
void D::h() { }

struct X {
  int ix;
  virtual void x();
};

// CHECK:      Vtable for 'Test1::E' (24 entries).
// CHECK-NEXT:    0 | vbase_offset (56)
// CHECK-NEXT:    1 | offset_to_top (0)
// CHECK-NEXT:    2 | Test1::E RTTI
// CHECK-NEXT:        -- (Test1::E, 0) vtable address --
// CHECK-NEXT:        -- (Test1::X, 0) vtable address --
// CHECK-NEXT:    3 | void Test1::X::x()
// CHECK-NEXT:    4 | void Test1::E::f()
// CHECK-NEXT:    5 | void Test1::E::h()
// CHECK-NEXT:    6 | vbase_offset (40)
// CHECK-NEXT:    7 | offset_to_top (-16)
// CHECK-NEXT:    8 | Test1::E RTTI
// CHECK-NEXT:        -- (Test1::B, 16) vtable address --
// CHECK-NEXT:        -- (Test1::D, 16) vtable address --
// CHECK-NEXT:    9 | void Test1::E::f()
// CHECK-NEXT:        [this adjustment: -16 non-virtual]
// CHECK-NEXT:   10 | void Test1::E::h()
// CHECK-NEXT:        [this adjustment: -16 non-virtual]
// CHECK-NEXT:   11 | vbase_offset (24)
// CHECK-NEXT:   12 | offset_to_top (-32)
// CHECK-NEXT:   13 | Test1::E RTTI
// CHECK-NEXT:        -- (Test1::C, 32) vtable address --
// CHECK-NEXT:   14 | void Test1::C::g()
// CHECK-NEXT:   15 | void Test1::E::h()
// CHECK-NEXT:        [this adjustment: -32 non-virtual]
// CHECK-NEXT:   16 | vcall_offset (-56)
// CHECK-NEXT:   17 | vcall_offset (-24)
// CHECK-NEXT:   18 | vcall_offset (-56)
// CHECK-NEXT:   19 | offset_to_top (-56)
// CHECK-NEXT:   20 | Test1::E RTTI
// CHECK-NEXT:        -- (Test1::A, 56) vtable address --
// CHECK-NEXT:   21 | void Test1::E::f()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-NEXT:   22 | void Test1::C::g()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]
// CHECK-NEXT:   23 | void Test1::E::h()
// CHECK-NEXT:        [this adjustment: 0 non-virtual, -40 vcall offset offset]
struct E : X, D {
  int ie;
  void f();
  void h ();
};
void E::f() { } 

}
