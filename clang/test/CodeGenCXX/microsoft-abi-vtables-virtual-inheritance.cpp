// RUN: %clang_cc1 -fno-rtti -emit-llvm -fdump-vtable-layouts %s -o - -cxx-abi microsoft -triple=i386-pc-win32 >%t 2>&1

// RUN: FileCheck --check-prefix=VTABLE-C %s < %t
// RUN: FileCheck --check-prefix=VTABLE-D %s < %t
// RUN: FileCheck --check-prefix=TEST1 %s < %t
// RUN: FileCheck --check-prefix=TEST2 %s < %t
// RUN: FileCheck --check-prefix=TEST3 %s < %t
// RUN: FileCheck --check-prefix=TEST4 %s < %t
// RUN: FileCheck --check-prefix=TEST5 %s < %t
// RUN: FileCheck --check-prefix=TEST6 %s < %t
// RUN: FileCheck --check-prefix=TEST7 %s < %t
// RUN: FileCheck --check-prefix=TEST8 %s < %t
// RUN: FileCheck --check-prefix=TEST9-Y %s < %t
// RUN: FileCheck --check-prefix=TEST9-Z %s < %t
// RUN: FileCheck --check-prefix=TEST9-W %s < %t
// RUN: FileCheck --check-prefix=TEST9-T %s < %t
// RUN: FileCheck --check-prefix=TEST10 %s < %t
// RUN: FileCheck --check-prefix=RET-W %s < %t
// RUN: FileCheck --check-prefix=RET-T %s < %t

struct Empty { };

struct A {
  virtual void f();
  virtual void z();  // Useful to check there are no thunks for f() when appropriate.
};

struct B {
  virtual void g();
};

struct C: virtual A {
  // VTABLE-C: VFTable for 'A' in 'C' (2 entries)
  // VTABLE-C-NEXT: 0 | void C::f()
  // VTABLE-C-NEXT: 1 | void A::z()

  // VTABLE-C: VFTable indices for 'C' (1 entries)
  // VTABLE-C-NEXT: vbtable index 1, vfptr at offset 0
  // VTABLE-C-NEXT: 0 | void C::f()

  ~C();  // Currently required to have correct record layout, see PR16406
  virtual void f();
};

C c;

struct D: virtual A {
  // VTABLE-D: VFTable for 'D' (1 entries).
  // VTABLE-D-NEXT: 0 | void D::h()

  // VTABLE-D: VFTable for 'A' in 'D' (2 entries).
  // VTABLE-D-NEXT: 0 | void D::f()
  // VTABLE-D-NEXT: 1 | void A::z()

  // VTABLE-D: VFTable indices for 'D' (2 entries).
  // VTABLE-D-NEXT: via vfptr at offset 0
  // VTABLE-D-NEXT: 0 | void D::h()
  // VTABLE-D-NEXT: via vbtable index 1, vfptr at offset 0
  // VTABLE-D-NEXT: 0 | void D::f()

  virtual void f();
  virtual void h();
};

void D::h() {}
D d;

namespace Test1 {

struct X { int x; };

// X and A get reordered in the layout since X doesn't have a vfptr while A has.
struct Y : X, A { };

struct Z : virtual Y {
  // TEST1: VFTable for 'A' in 'Test1::Y' in 'Test1::Z' (2 entries).
  // TEST1-NEXT: 0 | void A::f()
  // TEST1-NEXT: 1 | void A::z()

  // TEST1-NOT: VFTable indices for 'Test1::Z'
};

Z z;
}

namespace Test2 {

struct X: virtual A, virtual B {
  // TEST2: VFTable for 'Test2::X' (1 entries).
  // TEST2-NEXT: 0 | void Test2::X::h()

  // TEST2: VFTable for 'A' in 'Test2::X' (2 entries).
  // TEST2-NEXT: 0 | void A::f()
  // TEST2-NEXT: 1 | void A::z()

  // TEST2: VFTable for 'B' in 'Test2::X' (1 entries).
  // TEST2-NEXT: 0 | void B::g()

  // TEST2: VFTable indices for 'Test2::X' (1 entries).
  // TEST2-NEXT: 0 | void Test2::X::h()

  virtual void h();
};

X x;
}

namespace Test3 {

struct X : virtual A { };

struct Y: virtual X {
  // TEST3: VFTable for 'A' in 'Test3::X' in 'Test3::Y' (2 entries).
  // TEST3-NEXT: 0 | void A::f()
  // TEST3-NEXT: 1 | void A::z()

  // TEST3-NOT: VFTable indices for 'Test3::Y'
};

Y y;
}

namespace Test4 {

struct X: virtual C {
  // This one's interesting. C::f expects (A*) to be passed as 'this' and does
  // ECX-=4 to cast to (C*). In X, C and A vbases are reordered, so the thunk
  // should pass a pointer to the end of X in order
  // for ECX-=4 to point at the C part.

  // TEST4: VFTable for 'A' in 'C' in 'Test4::X' (2 entries).
  // TEST4-NEXT: 0 | void C::f()
  // TEST4-NEXT: [this adjustment: 12 non-virtual]
  // TEST4-NEXT: 1 | void A::z()

  // TEST4-NOT: VFTable indices for 'Test4::X'
};

X x;
}

namespace Test5 {

// New methods are added to the base's vftable.
struct X : A {
  virtual void g();
};

struct Y : virtual X {
  // TEST5: VFTable for 'Test5::Y' (1 entries).
  // TEST5-NEXT: 0 | void Test5::Y::h()

  // TEST5: VFTable for 'A' in 'Test5::X' in 'Test5::Y' (3 entries).
  // TEST5-NEXT: 0 | void A::f()
  // TEST5-NEXT: 1 | void A::z()
  // TEST5-NEXT: 2 | void Test5::X::g()

  // TEST5: VFTable indices for 'Test5::Y' (1 entries).
  // TEST5-NEXT: 0 | void Test5::Y::h()

  virtual void h();
};

Y y;
}

namespace Test6 {

struct X : A, virtual Empty {
  // TEST6: VFTable for 'A' in 'Test6::X' (2 entries).
  // TEST6-NEXT: 0 | void A::f()
  // TEST6-NEXT: 1 | void A::z()

  // TEST6-NOT: VFTable indices for 'Test6::X'
};

X x;
}

namespace Test7 {

struct X : C { };

struct Y : virtual X {
  // TEST7: VFTable for 'A' in 'C' in 'Test7::X' in 'Test7::Y' (2 entries).
  // TEST7-NEXT: 0 | void C::f()
  // TEST7-NEXT:     [this adjustment: 12 non-virtual]
  // TEST7-NEXT: 1 | void A::z()

  // TEST7: Thunks for 'void C::f()' (1 entry).
  // TEST7-NEXT: 0 | this adjustment: 12 non-virtual

  // TEST7-NOT: VFTable indices for 'Test7::Y'
};

Y y;
}

namespace Test8 {

// This is a typical diamond inheritance with a shared 'A' vbase.
struct X : D, C {
  // TEST8: VFTable for 'D' in 'Test8::X' (1 entries).
  // TEST8-NEXT: 0 | void D::h()

  // TEST8: VFTable for 'A' in 'D' in 'Test8::X' (2 entries).
  // TEST8-NEXT: 0 | void Test8::X::f()
  // TEST8-NEXT: 1 | void A::z()

  // TEST8: VFTable indices for 'Test8::X' (1 entries).
  // TEST8-NEXT: via vbtable index 1, vfptr at offset 0

  virtual void f();
};

X x;
}

namespace Test9 {

struct X : A { };

struct Y : virtual X {
  // TEST9-Y: VFTable for 'Test9::Y' (1 entries).
  // TEST9-Y-NEXT: 0 | void Test9::Y::h()

  // TEST9-Y: VFTable for 'A' in 'Test9::X' in 'Test9::Y' (2 entries).
  // TEST9-Y-NEXT: 0 | void A::f()
  // TEST9-Y-NEXT: 1 | void A::z()

  // TEST9-Y: VFTable indices for 'Test9::Y' (1 entries).
  // TEST9-Y-NEXT: 0 | void Test9::Y::h()

  virtual void h();
};

Y y;

struct Z : Y, virtual B {
  // TEST9-Z: VFTable for 'Test9::Y' in 'Test9::Z' (1 entries).
  // TEST9-Z-NEXT: 0 | void Test9::Y::h()

  // TEST9-Z: VFTable for 'A' in 'Test9::X' in 'Test9::Y' in 'Test9::Z' (2 entries).
  // TEST9-Z-NEXT: 0 | void A::f()
  // TEST9-Z-NEXT: 1 | void A::z()

  // TEST9-Z: VFTable for 'B' in 'Test9::Z' (1 entries).
  // TEST9-Z-NEXT: 0 | void B::g()

  // TEST9-Z-NOT: VFTable indices for 'Test9::Z'
};

Z z;

struct W : Z, D, virtual A, virtual B {
  // TEST9-W: VFTable for 'Test9::Y' in 'Test9::Z' in 'Test9::W' (1 entries).
  // TEST9-W-NEXT: 0 | void Test9::Y::h()

  // TEST9-W: VFTable for 'A' in 'Test9::X' in 'Test9::Y' in 'Test9::Z' in 'Test9::W' (2 entries).
  // TEST9-W-NEXT: 0 | void A::f()
  // TEST9-W-NEXT: 1 | void A::z()

  // TEST9-W: VFTable for 'B' in 'Test9::Z' in 'Test9::W' (1 entries).
  // TEST9-W-NEXT: 0 | void B::g()

  // TEST9-W: VFTable for 'D' in 'Test9::W' (1 entries).
  // TEST9-W-NEXT: 0 | void D::h()

  // TEST9-W: VFTable for 'A' in 'D' in 'Test9::W' (2 entries).
  // TEST9-W-NEXT: 0 | void D::f()
  // TEST9-W-NEXT:     [this adjustment: -8 non-virtual]
  // TEST9-W-NEXT: 1 | void A::z()

  // TEST9-W: Thunks for 'void D::f()' (1 entry).
  // TEST9-W-NEXT: 0 | this adjustment: -8 non-virtual

  // TEST9-W-NOT: VFTable indices for 'Test9::W'
};

W w;

struct T : Z, D, virtual A, virtual B {
  ~T();  // Currently required to have correct record layout, see PR16406

  // TEST9-T: VFTable for 'Test9::Y' in 'Test9::Z' in 'Test9::T' (1 entries).
  // TEST9-T-NEXT: 0 | void Test9::T::h()

  // TEST9-T: VFTable for 'A' in 'Test9::X' in 'Test9::Y' in 'Test9::Z' in 'Test9::T' (2 entries).
  // TEST9-T-NEXT: 0 | void Test9::T::f()
  // TEST9-T-NEXT: 1 | void Test9::T::z()

  // TEST9-T: VFTable for 'B' in 'Test9::Z' in 'Test9::T' (1 entries).
  // TEST9-T-NEXT: 0 | void Test9::T::g()

  // TEST9-T: VFTable for 'D' in 'Test9::T' (1 entries).
  // TEST9-T-NEXT: 0 | void Test9::T::h()
  // TEST9-T-NEXT:     [this adjustment: -8 non-virtual]

  // TEST9-T: Thunks for 'void Test9::T::h()' (1 entry).
  // TEST9-T-NEXT: 0 | this adjustment: -8 non-virtual

  // TEST9-T: VFTable for 'A' in 'D' in 'Test9::T' (2 entries).
  // TEST9-T-NEXT: 0 | void Test9::T::f()
  // TEST9-T-NEXT:     [this adjustment: -16 non-virtual]
  // TEST9-T-NEXT: 1 | void Test9::T::z()
  // TEST9-T-NEXT:     [this adjustment: -16 non-virtual]

  // TEST9-T: Thunks for 'void Test9::T::f()' (1 entry).
  // TEST9-T-NEXT: 0 | this adjustment: -16 non-virtual

  // TEST9-T: Thunks for 'void Test9::T::z()' (1 entry).
  // TEST9-T-NEXT: 0 | this adjustment: -16 non-virtual

  // TEST9-T: VFTable indices for 'Test9::T' (4 entries).
  // TEST9-T-NEXT: via vfptr at offset 0
  // TEST9-T-NEXT: 0 | void Test9::T::h()
  // TEST9-T-NEXT: via vbtable index 1, vfptr at offset 0
  // TEST9-T-NEXT: 0 | void Test9::T::f()
  // TEST9-T-NEXT: 1 | void Test9::T::z()
  // TEST9-T-NEXT: via vbtable index 2, vfptr at offset 0
  // TEST9-T-NEXT: 0 | void Test9::T::g()

  virtual void f();
  virtual void g();
  virtual void h();
  virtual void z();
};

T t;
}

namespace Test10 {
struct X : virtual C, virtual A {
  // TEST10: VFTable for 'A' in 'C' in 'Test10::X' (2 entries).
  // TEST10-NEXT: 0 | void Test10::X::f()
  // TEST10-NEXT: 1 | void A::z()

  // TEST10: VFTable indices for 'Test10::X' (1 entries).
  // TEST10-NEXT: via vbtable index 1, vfptr at offset 0
  // TEST10-NEXT: 0 | void Test10::X::f()
  virtual void f();
};

void X::f() {}
X x;
}

namespace return_adjustment {

struct X : virtual A {
  virtual void f();
};

struct Y : virtual A, virtual X {
  virtual void f();
};

struct Z {
  virtual A* foo();
};

struct W : Z {
  // RET-W: VFTable for 'return_adjustment::Z' in 'return_adjustment::W' (2 entries).
  // RET-W-NEXT: 0 | return_adjustment::X *return_adjustment::W::foo()
  // RET-W-NEXT:     [return adjustment: vbase #1, 0 non-virtual]
  // RET-W-NEXT: 1 | return_adjustment::X *return_adjustment::W::foo()

  // RET-W: VFTable indices for 'return_adjustment::W' (1 entries).
  // RET-W-NEXT: 1 | return_adjustment::X *return_adjustment::W::foo()

  virtual X* foo();
};

W y;

struct T : W {
  // RET-T: VFTable for 'return_adjustment::Z' in 'return_adjustment::W' in 'return_adjustment::T' (3 entries).
  // RET-T-NEXT: 0 | return_adjustment::Y *return_adjustment::T::foo()
  // RET-T-NEXT:     [return adjustment: vbase #1, 0 non-virtual]
  // RET-T-NEXT: 1 | return_adjustment::Y *return_adjustment::T::foo()
  // RET-T-NEXT:     [return adjustment: vbase #2, 0 non-virtual]
  // RET-T-NEXT: 2 | return_adjustment::Y *return_adjustment::T::foo()

  // RET-T: VFTable indices for 'return_adjustment::T' (1 entries).
  // RET-T-NEXT: 2 | return_adjustment::Y *return_adjustment::T::foo()

  virtual Y* foo();
};

T t;
}
