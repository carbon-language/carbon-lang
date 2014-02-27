// RUN: %clang_cc1 -fno-rtti -emit-llvm -o %t.ll -fdump-vtable-layouts %s -triple=i386-pc-win32 >%t

// RUN: FileCheck --check-prefix=VTABLE-C %s < %t
// RUN: FileCheck --check-prefix=VTABLE-D %s < %t
// RUN: FileCheck --check-prefix=TEST1 %s < %t
// RUN: FileCheck --check-prefix=TEST2 %s < %t
// RUN: FileCheck --check-prefix=TEST3 %s < %t
// RUN: FileCheck --check-prefix=TEST4 %s < %t
// RUN: FileCheck --check-prefix=TEST5 %s < %t
// RUN: FileCheck --check-prefix=TEST6 %s < %t
// RUN: FileCheck --check-prefix=TEST7 %s < %t
// RUN: FileCheck --check-prefix=TEST8-X %s < %t
// RUN: FileCheck --check-prefix=TEST8-Z %s < %t
// RUN: FileCheck --check-prefix=TEST9-Y %s < %t
// RUN: FileCheck --check-prefix=TEST9-Z %s < %t
// RUN: FileCheck --check-prefix=TEST9-W %s < %t
// RUN: FileCheck --check-prefix=TEST9-T %s < %t
// RUN: FileCheck --check-prefix=TEST10 %s < %t
// RUN: FileCheck --check-prefix=VDTORS-Y %s < %t
// RUN: FileCheck --check-prefix=VDTORS-U %s < %t
// RUN: FileCheck --check-prefix=VDTORS-V %s < %t
// RUN: FileCheck --check-prefix=VDTORS-P %s < %t
// RUN: FileCheck --check-prefix=RET-W %s < %t
// RUN: FileCheck --check-prefix=RET-T %s < %t
// RUN: FileCheck --check-prefix=RET-V %s < %t

// RUN: FileCheck --check-prefix=MANGLING %s < %t.ll

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

  // MANGLING-DAG: @"\01??_7C@@6B@"

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

  // MANGLING-DAG: @"\01??_7D@@6B0@@"
  // MANGLING-DAG: @"\01??_7D@@6BA@@@"

  virtual void f();
  virtual void h();
};

void D::h() {}
D d;

namespace Test1 {

struct X { int x; };

// X and A get reordered in the layout since X doesn't have a vfptr while A has.
struct Y : X, A { };
// MANGLING-DAG: @"\01??_7Y@Test1@@6B@"

struct Z : virtual Y {
  // TEST1: VFTable for 'A' in 'Test1::Y' in 'Test1::Z' (2 entries).
  // TEST1-NEXT: 0 | void A::f()
  // TEST1-NEXT: 1 | void A::z()

  // TEST1-NOT: VFTable indices for 'Test1::Z'

  // MANGLING-DAG: @"\01??_7Z@Test1@@6B@"
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

  // MANGLING-DAG: @"\01??_7X@Test2@@6B01@@"
  // MANGLING-DAG: @"\01??_7X@Test2@@6BA@@@"
  // MANGLING-DAG: @"\01??_7X@Test2@@6BB@@@"

  virtual void h();
};

X x;
}

namespace Test3 {

struct X : virtual A {
  // MANGLING-DAG: @"\01??_7X@Test3@@6B@"
};

struct Y: virtual X {
  // TEST3: VFTable for 'A' in 'Test3::X' in 'Test3::Y' (2 entries).
  // TEST3-NEXT: 0 | void A::f()
  // TEST3-NEXT: 1 | void A::z()

  // TEST3-NOT: VFTable indices for 'Test3::Y'

  // MANGLING-DAG: @"\01??_7Y@Test3@@6B@"
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
  // TEST4-NEXT:     [this adjustment: 8 non-virtual]
  // TEST4-NEXT: 1 | void A::z()

  // TEST4-NOT: VFTable indices for 'Test4::X'

  // MANGLING-DAG: @"\01??_7X@Test4@@6B@"

  // Also check the mangling of the thunk.
  // MANGLING-DAG: define weak x86_thiscallcc void @"\01?f@C@@WPPPPPPPI@AEXXZ"
};

X x;
}

namespace Test5 {

// New methods are added to the base's vftable.
struct X : A {
  // MANGLING-DAG: @"\01??_7X@Test5@@6B@"
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

  // MANGLING-DAG: @"\01??_7Y@Test5@@6B01@@"
  // MANGLING-DAG: @"\01??_7Y@Test5@@6BX@1@@"

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

  // MANGLING-DAG: @"\01??_7X@Test6@@6B@"
};

X x;
}

namespace Test7 {

struct X : C {
  // MANGLING-DAG: @"\01??_7X@Test7@@6B@"
};

struct Y : virtual X {
  // TEST7: VFTable for 'A' in 'C' in 'Test7::X' in 'Test7::Y' (2 entries).
  // TEST7-NEXT: 0 | void C::f()
  // TEST7-NEXT:     [this adjustment: 8 non-virtual]
  // TEST7-NEXT: 1 | void A::z()

  // TEST7: Thunks for 'void C::f()' (1 entry).
  // TEST7-NEXT: 0 | [this adjustment: 8 non-virtual]

  // TEST7-NOT: VFTable indices for 'Test7::Y'

  // MANGLING-DAG: @"\01??_7Y@Test7@@6B@"
};

Y y;
}

namespace Test8 {

// This is a typical diamond inheritance with a shared 'A' vbase.
struct X : D, C {
  // TEST8-X: VFTable for 'D' in 'Test8::X' (1 entries).
  // TEST8-X-NEXT: 0 | void D::h()

  // TEST8-X: VFTable for 'A' in 'D' in 'Test8::X' (2 entries).
  // TEST8-X-NEXT: 0 | void Test8::X::f()
  // TEST8-X-NEXT: 1 | void A::z()

  // TEST8-X: VFTable indices for 'Test8::X' (1 entries).
  // TEST8-X-NEXT: via vbtable index 1, vfptr at offset 0
  // TEST8-X-NEXT: 0 | void Test8::X::f()

  // MANGLING-DAG: @"\01??_7X@Test8@@6BA@@@"
  // MANGLING-DAG: @"\01??_7X@Test8@@6BD@@@"

  virtual void f();
};

X x;

// Another diamond inheritance which led to AST crashes.
struct Y : virtual A {};

class Z : Y, C {
  // TEST8-Z: VFTable for 'A' in 'Test8::Y' in 'Test8::Z' (2 entries).
  // TEST8-Z-NEXT: 0 | void Test8::Z::f()
  // TEST8-Z-NEXT: 1 | void A::z()

  // TEST8-Z: VFTable indices for 'Test8::Z' (1 entries).
  // TEST8-Z-NEXT: via vbtable index 1, vfptr at offset 0
  // TEST8-Z-NEXT: 0 | void Test8::Z::f()
  virtual void f();
};
Z z;
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

  // MANGLING-DAG: @"\01??_7Y@Test9@@6B01@@"
  // MANGLING-DAG: @"\01??_7Y@Test9@@6BX@1@@"

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

  // MANGLING-DAG: @"\01??_7Z@Test9@@6BX@1@@"
  // MANGLING-DAG: @"\01??_7Z@Test9@@6BY@1@@"

  // MANGLING-DAG: @"\01??_7Z@Test9@@6B@"
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
  // TEST9-W-NEXT: 0 | [this adjustment: -8 non-virtual]

  // TEST9-W-NOT: VFTable indices for 'Test9::W'

  // MANGLING-DAG: @"\01??_7W@Test9@@6BA@@@"
  // MANGLING-DAG: @"\01??_7W@Test9@@6BD@@@"
  // MANGLING-DAG: @"\01??_7W@Test9@@6BX@1@@"

  // MANGLING-DAG: @"\01??_7W@Test9@@6B@"
  // MANGLING-DAG: @"\01??_7W@Test9@@6BY@1@@"
};

W w;

struct T : Z, D, virtual A, virtual B {
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
  // TEST9-T-NEXT: 0 | [this adjustment: -8 non-virtual]

  // TEST9-T: VFTable for 'A' in 'D' in 'Test9::T' (2 entries).
  // TEST9-T-NEXT: 0 | void Test9::T::f()
  // TEST9-T-NEXT:     [this adjustment: -8 non-virtual]
  // TEST9-T-NEXT: 1 | void Test9::T::z()
  // TEST9-T-NEXT:     [this adjustment: -8 non-virtual]

  // TEST9-T: Thunks for 'void Test9::T::f()' (1 entry).
  // TEST9-T-NEXT: 0 | [this adjustment: -8 non-virtual]

  // TEST9-T: Thunks for 'void Test9::T::z()' (1 entry).
  // TEST9-T-NEXT: 0 | [this adjustment: -8 non-virtual]

  // TEST9-T: VFTable indices for 'Test9::T' (4 entries).
  // TEST9-T-NEXT: via vfptr at offset 0
  // TEST9-T-NEXT: 0 | void Test9::T::h()
  // TEST9-T-NEXT: via vbtable index 1, vfptr at offset 0
  // TEST9-T-NEXT: 0 | void Test9::T::f()
  // TEST9-T-NEXT: 1 | void Test9::T::z()
  // TEST9-T-NEXT: via vbtable index 2, vfptr at offset 0
  // TEST9-T-NEXT: 0 | void Test9::T::g()

  // MANGLING-DAG: @"\01??_7T@Test9@@6BA@@@"
  // MANGLING-DAG: @"\01??_7T@Test9@@6BD@@@"
  // MANGLING-DAG: @"\01??_7T@Test9@@6BX@1@@"

  // MANGLING-DAG: @"\01??_7T@Test9@@6B@"
  // MANGLING-DAG: @"\01??_7T@Test9@@6BY@1@@"

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

namespace Test11 {
struct X : virtual A {};
struct Y { virtual void g(); };

struct Z : virtual X, Y {
  // MANGLING-DAG: @"\01??_7Z@Test11@@6BY@1@@"
  // MANGLING-DAG: @"\01??_7Z@Test11@@6BX@1@@"
};

Z z;

struct W : virtual X, A {};

// PR17748 FIXME this one hits UNREACHABLE:
// W w;
}

namespace vdtors {
struct X {
  virtual ~X();
  virtual void zzz();
};

struct Y : virtual X {
  // VDTORS-Y: VFTable for 'vdtors::X' in 'vdtors::Y' (2 entries).
  // VDTORS-Y-NEXT: 0 | vdtors::Y::~Y() [scalar deleting]
  // VDTORS-Y-NEXT: 1 | void vdtors::X::zzz()

  // VDTORS-Y-NOT: Thunks for 'vdtors::Y::~Y()'
  virtual ~Y();
};

Y y;

struct Z {
  virtual void z();
};

struct W : Z, X {
  // Implicit virtual dtor.
};

struct U : virtual W {
  // VDTORS-U: VFTable for 'vdtors::Z' in 'vdtors::W' in 'vdtors::U' (1 entries).
  // VDTORS-U-NEXT: 0 | void vdtors::Z::z()

  // VDTORS-U: VFTable for 'vdtors::X' in 'vdtors::W' in 'vdtors::U' (2 entries).
  // VDTORS-U-NEXT: 0 | vdtors::U::~U() [scalar deleting]
  // VDTORS-U-NEXT:     [this adjustment: -4 non-virtual]
  // VDTORS-U-NEXT: 1 | void vdtors::X::zzz()

  // VDTORS-U: Thunks for 'vdtors::W::~W()' (1 entry).
  // VDTORS-U-NEXT: 0 | [this adjustment: -4 non-virtual]

  // VDTORS-U: VFTable indices for 'vdtors::U' (1 entries).
  // VDTORS-U-NEXT: -- accessible via vbtable index 1, vfptr at offset 4 --
  // VDTORS-U-NEXT: 0 | vdtors::U::~U() [scalar deleting]
  virtual ~U();
};

U u;

struct V : virtual W {
  // VDTORS-V: VFTable for 'vdtors::Z' in 'vdtors::W' in 'vdtors::V' (1 entries).
  // VDTORS-V-NEXT: 0 | void vdtors::Z::z()

  // VDTORS-V: VFTable for 'vdtors::X' in 'vdtors::W' in 'vdtors::V' (2 entries).
  // VDTORS-V-NEXT: 0 | vdtors::V::~V() [scalar deleting]
  // VDTORS-V-NEXT:     [this adjustment: -4 non-virtual]
  // VDTORS-V-NEXT: 1 | void vdtors::X::zzz()

  // VDTORS-V: Thunks for 'vdtors::W::~W()' (1 entry).
  // VDTORS-V-NEXT: 0 | [this adjustment: -4 non-virtual]

  // VDTORS-V: VFTable indices for 'vdtors::V' (1 entries).
  // VDTORS-V-NEXT: -- accessible via vbtable index 1, vfptr at offset 4 --
  // VDTORS-V-NEXT: 0 | vdtors::V::~V() [scalar deleting]
};

V v;

struct T : virtual X {
  virtual ~T();
};

struct P : T, Y {
  // VDTORS-P: VFTable for 'vdtors::X' in 'vdtors::T' in 'vdtors::P' (2 entries).
  // VDTORS-P-NEXT: 0 | vdtors::P::~P() [scalar deleting]
  // VDTORS-P-NEXT: 1 | void vdtors::X::zzz()

  // VDTORS-P-NOT: Thunks for 'vdtors::P::~P()'
  virtual ~P();
};

P p;

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

struct U : virtual A {
  virtual void g();  // adds a vfptr
};

struct V : Z {
  // RET-V: VFTable for 'return_adjustment::Z' in 'return_adjustment::V' (2 entries).
  // RET-V-NEXT: 0 | return_adjustment::U *return_adjustment::V::foo()
  // RET-V-NEXT:     [return adjustment: vbptr at offset 4, vbase #1, 0 non-virtual]
  // RET-V-NEXT: 1 | return_adjustment::U *return_adjustment::V::foo()

  // RET-V: VFTable indices for 'return_adjustment::V' (1 entries).
  // RET-V-NEXT: 1 | return_adjustment::U *return_adjustment::V::foo()

  virtual U* foo();
};

V v;
}

namespace pr17748 {
struct A {
  virtual void f() {}
};

struct B : virtual A {
  B() {}
};

struct C : virtual B, A {
  C() {}
};
C c;

// MANGLING-DAG: @"\01??_7A@pr17748@@6B@"
// MANGLING-DAG: @"\01??_7B@pr17748@@6B@"
// MANGLING-DAG: @"\01??_7C@pr17748@@6BA@1@@"
// MANGLING-DAG: @"\01??_7C@pr17748@@6BB@1@@"
}
