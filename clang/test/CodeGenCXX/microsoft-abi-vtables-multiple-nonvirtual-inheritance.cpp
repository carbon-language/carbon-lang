// RUN: %clang_cc1 %s -fno-rtti -cxx-abi microsoft -triple=i386-pc-win32 -emit-llvm -o %t.ll -fdump-vtable-layouts >%t

// RUN: FileCheck --check-prefix=NO-THUNKS-Test1 %s < %t
// RUN: FileCheck --check-prefix=NO-THUNKS-Test2 %s < %t
// RUN: FileCheck --check-prefix=NO-THUNKS-Test3 %s < %t
// RUN: FileCheck --check-prefix=NO-THUNKS-Test4 %s < %t
// RUN: FileCheck --check-prefix=NO-THUNKS-Test5 %s < %t
// RUN: FileCheck --check-prefix=NO-THUNKS-Test6 %s < %t
// RUN: FileCheck --check-prefix=NO-THUNKS-Test7 %s < %t
// RUN: FileCheck --check-prefix=NO-THUNKS-Test8 %s < %t
// RUN: FileCheck --check-prefix=NO-THUNKS-Test9 %s < %t
// RUN: FileCheck --check-prefix=PURE-VIRTUAL-Test1 %s < %t
// RUN: FileCheck --check-prefix=THIS-THUNKS-Test1 %s < %t
// RUN: FileCheck --check-prefix=THIS-THUNKS-Test2 %s < %t
// RUN: FileCheck --check-prefix=THIS-THUNKS-Test3 %s < %t
// RUN: FileCheck --check-prefix=VDTOR-THUNKS-Test3 %s < %t
// RUN: FileCheck --check-prefix=VDTOR-THUNKS-Test5 %s < %t
// RUN: FileCheck --check-prefix=VDTOR-THUNKS-Test6 %s < %t
// RUN: FileCheck --check-prefix=VDTOR-THUNKS-Test7 %s < %t
// RUN: FileCheck --check-prefix=RET-THUNKS-Test1 %s < %t
// RUN: FileCheck --check-prefix=RET-THUNKS-Test2 %s < %t
// RUN: FileCheck --check-prefix=RET-THUNKS-Test3 %s < %t
// RUN: FileCheck --check-prefix=RET-THUNKS-Test4 %s < %t
// RUN: FileCheck --check-prefix=RET-THUNKS-Test5 %s < %t
// RUN: FileCheck --check-prefix=RET-THUNKS-Test6 %s < %t

// RUN: FileCheck --check-prefix=MANGLING %s < %t.ll

struct Empty {
  // Doesn't have a vftable!
};

struct A {
  virtual void f();
};

struct B {
  virtual void g();
  // Add an extra virtual method so it's easier to check for the absence of thunks.
  virtual void h();
};

struct C {
  virtual void g();  // Might "collide" with B::g if both are bases of some class.
};


namespace no_thunks {

struct Test1: A, B {
  // NO-THUNKS-Test1: VFTable for 'A' in 'no_thunks::Test1' (1 entries)
  // NO-THUNKS-Test1-NEXT: 0 | void no_thunks::Test1::f()

  // NO-THUNKS-Test1: VFTable for 'B' in 'no_thunks::Test1' (2 entries)
  // NO-THUNKS-Test1-NEXT: 0 | void B::g()
  // NO-THUNKS-Test1-NEXT: 1 | void B::h()

  // NO-THUNKS-Test1: VFTable indices for 'no_thunks::Test1' (1 entries)
  // NO-THUNKS-Test1-NEXT: 0 | void no_thunks::Test1::f()

  // MANGLING-DAG: @"\01??_7Test1@no_thunks@@6BA@@@"
  // MANGLING-DAG: @"\01??_7Test1@no_thunks@@6BB@@@"

  // Overrides only the left child's method (A::f), needs no thunks.
  virtual void f();
};

Test1 t1;

struct Test2: A, B {
  // NO-THUNKS-Test2: VFTable for 'A' in 'no_thunks::Test2' (1 entries)
  // NO-THUNKS-Test2-NEXT: 0 | void A::f()

  // NO-THUNKS-Test2: VFTable for 'B' in 'no_thunks::Test2' (2 entries)
  // NO-THUNKS-Test2-NEXT: 0 | void no_thunks::Test2::g()
  // NO-THUNKS-Test2-NEXT: 1 | void B::h()

  // NO-THUNKS-Test2: VFTable indices for 'no_thunks::Test2' (1 entries).
  // NO-THUNKS-Test2-NEXT: via vfptr at offset 4
  // NO-THUNKS-Test2-NEXT: 0 | void no_thunks::Test2::g()

  // Overrides only the right child's method (B::g), needs this adjustment but
  // not thunks.
  virtual void g();
};

Test2 t2;

struct Test3: A, B {
  // NO-THUNKS-Test3: VFTable for 'A' in 'no_thunks::Test3' (2 entries)
  // NO-THUNKS-Test3-NEXT: 0 | void A::f()
  // NO-THUNKS-Test3-NEXT: 1 | void no_thunks::Test3::i()

  // NO-THUNKS-Test3: VFTable for 'B' in 'no_thunks::Test3' (2 entries)
  // NO-THUNKS-Test3-NEXT: 0 | void B::g()
  // NO-THUNKS-Test3-NEXT: 1 | void B::h()

  // NO-THUNKS-Test3: VFTable indices for 'no_thunks::Test3' (1 entries).
  // NO-THUNKS-Test3-NEXT: 1 | void no_thunks::Test3::i()

  // Only adds a new method.
  virtual void i();
};

Test3 t3;

// Only the right base has a vftable, so it's laid out before the left one!
struct Test4 : Empty, A {
  // NO-THUNKS-Test4: VFTable for 'A' in 'no_thunks::Test4' (1 entries)
  // NO-THUNKS-Test4-NEXT: 0 | void no_thunks::Test4::f()

  // NO-THUNKS-Test4: VFTable indices for 'no_thunks::Test4' (1 entries).
  // NO-THUNKS-Test4-NEXT: 0 | void no_thunks::Test4::f()

  // MANGLING-DAG: @"\01??_7Test4@no_thunks@@6B@"

  virtual void f();
};

Test4 t4;

// 2-level structure with repeating subobject types, but no thunks needed.
struct Test5: Test1, Test2 {
  // NO-THUNKS-Test5: VFTable for 'A' in 'no_thunks::Test1' in 'no_thunks::Test5' (2 entries)
  // NO-THUNKS-Test5-NEXT: 0 | void no_thunks::Test1::f()
  // NO-THUNKS-Test5-NEXT: 1 | void no_thunks::Test5::z()

  // NO-THUNKS-Test5: VFTable for 'B' in 'no_thunks::Test1' in 'no_thunks::Test5' (2 entries)
  // NO-THUNKS-Test5-NEXT: 0 | void B::g()
  // NO-THUNKS-Test5-NEXT: 1 | void B::h()

  // NO-THUNKS-Test5: VFTable for 'A' in 'no_thunks::Test2' in 'no_thunks::Test5' (1 entries)
  // NO-THUNKS-Test5-NEXT: 0 | void A::f()

  // NO-THUNKS-Test5: VFTable for 'B' in 'no_thunks::Test2' in 'no_thunks::Test5' (2 entries)
  // NO-THUNKS-Test5-NEXT: 0 | void no_thunks::Test2::g()
  // NO-THUNKS-Test5-NEXT: 1 | void B::h()

  // NO-THUNKS-Test5: VFTable indices for 'no_thunks::Test5' (1 entries).
  // NO-THUNKS-Test5-NEXT: 1 | void no_thunks::Test5::z()

  // MANGLING-DAG: @"\01??_7Test5@no_thunks@@6BA@@Test1@1@@"
  // MANGLING-DAG: @"\01??_7Test5@no_thunks@@6BA@@Test2@1@@"
  // MANGLING-DAG: @"\01??_7Test5@no_thunks@@6BB@@Test1@1@@"
  // MANGLING-DAG: @"\01??_7Test5@no_thunks@@6BB@@Test2@1@@"

  virtual void z();
};

Test5 t5;

struct Test6: Test1 {
  // NO-THUNKS-Test6: VFTable for 'A' in 'no_thunks::Test1' in 'no_thunks::Test6' (1 entries).
  // NO-THUNKS-Test6-NEXT: 0 | void no_thunks::Test6::f()

  // NO-THUNKS-Test6: VFTable for 'B' in 'no_thunks::Test1' in 'no_thunks::Test6' (2 entries).
  // NO-THUNKS-Test6-NEXT: 0 | void B::g()
  // NO-THUNKS-Test6-NEXT: 1 | void B::h()

  // NO-THUNKS-Test6: VFTable indices for 'no_thunks::Test6' (1 entries).
  // NO-THUNKS-Test6-NEXT: 0 | void no_thunks::Test6::f()

  // MANGLING-DAG: @"\01??_7Test6@no_thunks@@6BA@@@"
  // MANGLING-DAG: @"\01??_7Test6@no_thunks@@6BB@@@"

  // Overrides both no_thunks::Test1::f and A::f.
  virtual void f();
};

Test6 t6;

struct Test7: Test2 {
  // NO-THUNKS-Test7: VFTable for 'A' in 'no_thunks::Test2' in 'no_thunks::Test7' (1 entries).
  // NO-THUNKS-Test7-NEXT: 0 | void A::f()

  // NO-THUNKS-Test7: VFTable for 'B' in 'no_thunks::Test2' in 'no_thunks::Test7' (2 entries).
  // NO-THUNKS-Test7-NEXT: 0 | void no_thunks::Test7::g()
  // NO-THUNKS-Test7-NEXT: 1 | void B::h()

  // NO-THUNKS-Test7: VFTable indices for 'no_thunks::Test7' (1 entries).
  // NO-THUNKS-Test7-NEXT: via vfptr at offset 4
  // NO-THUNKS-Test7-NEXT: 0 | void no_thunks::Test7::g()

  // Overrides both no_thunks::Test2::g and B::g.
  virtual void g();
};

Test7 t7;

struct Test8: Test3 {
  // NO-THUNKS-Test8: VFTable for 'A' in 'no_thunks::Test3' in 'no_thunks::Test8' (2 entries).
  // NO-THUNKS-Test8-NEXT: 0 | void A::f()
  // NO-THUNKS-Test8-NEXT: 1 | void no_thunks::Test3::i()

  // NO-THUNKS-Test8: VFTable for 'B' in 'no_thunks::Test3' in 'no_thunks::Test8' (2 entries).
  // NO-THUNKS-Test8-NEXT: 0 | void no_thunks::Test8::g()
  // NO-THUNKS-Test8-NEXT: 1 | void B::h()

  // NO-THUNKS-Test8: VFTable indices for 'no_thunks::Test8' (1 entries).
  // NO-THUNKS-Test8-NEXT: via vfptr at offset 4
  // NO-THUNKS-Test8-NEXT: 0 | void no_thunks::Test8::g()

  // Overrides grandparent's B::g.
  virtual void g();
};

Test8 t8;

struct D : A {
  virtual void g();
};

// Repeating subobject.
struct Test9: A, D {
  // NO-THUNKS-Test9: VFTable for 'A' in 'no_thunks::Test9' (2 entries).
  // NO-THUNKS-Test9-NEXT: 0 | void A::f()
  // NO-THUNKS-Test9-NEXT: 1 | void no_thunks::Test9::h()

  // NO-THUNKS-Test9: VFTable for 'A' in 'no_thunks::D' in 'no_thunks::Test9' (2 entries).
  // NO-THUNKS-Test9-NEXT: 0 | void A::f()
  // NO-THUNKS-Test9-NEXT: 1 | void no_thunks::D::g()

  // NO-THUNKS-Test9: VFTable indices for 'no_thunks::Test9' (1 entries).
  // NO-THUNKS-Test9-NEXT: 1 | void no_thunks::Test9::h()

  // MANGLING-DAG: @"\01??_7Test9@no_thunks@@6BA@@@"
  // MANGLING-DAG: @"\01??_7Test9@no_thunks@@6BD@1@@"

  virtual void h();
};

Test9 t9;
}

namespace pure_virtual {
struct D {
  virtual void g() = 0;
  virtual void h();
};


struct Test1: A, D {
  // PURE-VIRTUAL-Test1: VFTable for 'A' in 'pure_virtual::Test1' (1 entries)
  // PURE-VIRTUAL-Test1-NEXT: 0 | void A::f()

  // PURE-VIRTUAL-Test1: VFTable for 'pure_virtual::D' in 'pure_virtual::Test1' (2 entries)
  // PURE-VIRTUAL-Test1-NEXT: 0 | void pure_virtual::Test1::g()
  // PURE-VIRTUAL-Test1-NEXT: 1 | void pure_virtual::D::h()

  // PURE-VIRTUAL-Test1: VFTable indices for 'pure_virtual::Test1' (1 entries).
  // PURE-VIRTUAL-Test1-NEXT: via vfptr at offset 4
  // PURE-VIRTUAL-Test1-NEXT: 0 | void pure_virtual::Test1::g()

  // MANGLING-DAG: @"\01??_7Test1@pure_virtual@@6BA@@@"
  // MANGLING-DAG: @"\01??_7Test1@pure_virtual@@6BD@1@@"

  // Overrides only the right child's method (pure_virtual::D::g), needs this adjustment but
  // not thunks.
  virtual void g();
};

Test1 t1;
}

namespace this_adjustment {

// Overrides methods of two bases at the same time, thus needing thunks.
struct Test1 : B, C {
  // THIS-THUNKS-Test1: VFTable for 'B' in 'this_adjustment::Test1' (2 entries).
  // THIS-THUNKS-Test1-NEXT: 0 | void this_adjustment::Test1::g()
  // THIS-THUNKS-Test1-NEXT: 1 | void B::h()

  // THIS-THUNKS-Test1: VFTable for 'C' in 'this_adjustment::Test1' (1 entries).
  // THIS-THUNKS-Test1-NEXT: 0 | void this_adjustment::Test1::g()
  // THIS-THUNKS-Test1-NEXT:     [this adjustment: -4 non-virtual]

  // THIS-THUNKS-Test1: Thunks for 'void this_adjustment::Test1::g()' (1 entry).
  // THIS-THUNKS-Test1-NEXT: 0 | [this adjustment: -4 non-virtual]

  // THIS-THUNKS-Test1: VFTable indices for 'this_adjustment::Test1' (1 entries).
  // THIS-THUNKS-Test1-NEXT: 0 | void this_adjustment::Test1::g()

  // MANGLING-DAG: @"\01??_7Test1@this_adjustment@@6BB@@@"
  // MANGLING-DAG: @"\01??_7Test1@this_adjustment@@6BC@@@"

  virtual void g();
};

Test1 t1;

struct Test2 : A, B, C {
  // THIS-THUNKS-Test2: VFTable for 'A' in 'this_adjustment::Test2' (1 entries).
  // THIS-THUNKS-Test2-NEXT: 0 | void A::f()

  // THIS-THUNKS-Test2: VFTable for 'B' in 'this_adjustment::Test2' (2 entries).
  // THIS-THUNKS-Test2-NEXT: 0 | void this_adjustment::Test2::g()
  // THIS-THUNKS-Test2-NEXT: 1 | void B::h()

  // THIS-THUNKS-Test2: VFTable for 'C' in 'this_adjustment::Test2' (1 entries).
  // THIS-THUNKS-Test2-NEXT: 0 | void this_adjustment::Test2::g()
  // THIS-THUNKS-Test2-NEXT:     [this adjustment: -4 non-virtual]

  // THIS-THUNKS-Test2: Thunks for 'void this_adjustment::Test2::g()' (1 entry).
  // THIS-THUNKS-Test2-NEXT: 0 | [this adjustment: -4 non-virtual]

  // THIS-THUNKS-Test2: VFTable indices for 'this_adjustment::Test2' (1 entries).
  // THIS-THUNKS-Test2-NEXT: via vfptr at offset 4
  // THIS-THUNKS-Test2-NEXT: 0 | void this_adjustment::Test2::g()

  // MANGLING-DAG: @"\01??_7Test2@this_adjustment@@6BA@@@"
  // MANGLING-DAG: @"\01??_7Test2@this_adjustment@@6BB@@@"
  // MANGLING-DAG: @"\01??_7Test2@this_adjustment@@6BC@@@"

  virtual void g();
};

Test2 t2;

// Overrides methods of two bases at the same time, thus needing thunks.
struct Test3: no_thunks::Test1, no_thunks::Test2 {
  // THIS-THUNKS-Test3: VFTable for 'A' in 'no_thunks::Test1' in 'this_adjustment::Test3' (1 entries).
  // THIS-THUNKS-Test3-NEXT: 0 | void this_adjustment::Test3::f()

  // THIS-THUNKS-Test3: VFTable for 'B' in 'no_thunks::Test1' in 'this_adjustment::Test3' (2 entries).
  // THIS-THUNKS-Test3-NEXT: 0 | void this_adjustment::Test3::g()
  // THIS-THUNKS-Test3-NEXT: 1 | void B::h()

  // THIS-THUNKS-Test3: VFTable for 'A' in 'no_thunks::Test2' in 'this_adjustment::Test3' (1 entries).
  // THIS-THUNKS-Test3-NEXT: 0 | void this_adjustment::Test3::f()
  // THIS-THUNKS-Test3-NEXT:     [this adjustment: -8 non-virtual]

  // THIS-THUNKS-Test3: Thunks for 'void this_adjustment::Test3::f()' (1 entry).
  // THIS-THUNKS-Test3-NEXT: 0 | [this adjustment: -8 non-virtual]

  // THIS-THUNKS-Test3: VFTable for 'B' in 'no_thunks::Test2' in 'this_adjustment::Test3' (2 entries).
  // THIS-THUNKS-Test3-NEXT: 0 | void this_adjustment::Test3::g()
  // THIS-THUNKS-Test3-NEXT:     [this adjustment: -8 non-virtual]
  // THIS-THUNKS-Test3-NEXT: 1 | void B::h()

  // THIS-THUNKS-Test3: Thunks for 'void this_adjustment::Test3::g()' (1 entry).
  // THIS-THUNKS-Test3-NEXT: 0 | [this adjustment: -8 non-virtual]

  // THIS-THUNKS-Test3: VFTable indices for 'this_adjustment::Test3' (2 entries).
  // THIS-THUNKS-Test3-NEXT: via vfptr at offset 0
  // THIS-THUNKS-Test3-NEXT: 0 | void this_adjustment::Test3::f()
  // THIS-THUNKS-Test3-NEXT: via vfptr at offset 4
  // THIS-THUNKS-Test3-NEXT: 0 | void this_adjustment::Test3::g()

  virtual void f();
  virtual void g();
};

Test3 t3;
}

namespace vdtor {
struct Test1 {
  virtual ~Test1();
  virtual void z1();
};

struct Test2 {
  virtual ~Test2();
};

struct Test3 : Test1, Test2 {
  // VDTOR-THUNKS-Test3: VFTable for 'vdtor::Test1' in 'vdtor::Test3' (2 entries).
  // VDTOR-THUNKS-Test3-NEXT: 0 | vdtor::Test3::~Test3() [scalar deleting]
  // VDTOR-THUNKS-Test3-NEXT: 1 | void vdtor::Test1::z1()

  // VDTOR-THUNKS-Test3: VFTable for 'vdtor::Test2' in 'vdtor::Test3' (1 entries).
  // VDTOR-THUNKS-Test3-NEXT: 0 | vdtor::Test3::~Test3() [scalar deleting]
  // VDTOR-THUNKS-Test3-NEXT:     [this adjustment: -4 non-virtual]

  // VDTOR-THUNKS-Test3: Thunks for 'vdtor::Test3::~Test3()' (1 entry).
  // VDTOR-THUNKS-Test3-NEXT: 0 | [this adjustment: -4 non-virtual]

  // VDTOR-THUNKS-Test3: VFTable indices for 'vdtor::Test3' (1 entries).
  // VDTOR-THUNKS-Test3-NEXT: 0 | vdtor::Test3::~Test3() [scalar deleting]
  virtual ~Test3();
};

Test3 t3;

struct Test4 {
  // No virtual destructor here!
  virtual void z4();
};

struct Test5 : Test4, Test2 {
  // Implicit virtual dtor here!

  // VDTOR-THUNKS-Test5: VFTable for 'vdtor::Test4' in 'vdtor::Test5' (1 entries).
  // VDTOR-THUNKS-Test5-NEXT: 0 | void vdtor::Test4::z4()

  // VDTOR-THUNKS-Test5: VFTable for 'vdtor::Test2' in 'vdtor::Test5' (1 entries).
  // VDTOR-THUNKS-Test5-NEXT: 0 | vdtor::Test5::~Test5() [scalar deleting]
  // VDTOR-THUNKS-Test5-NEXT:     [this adjustment: -4 non-virtual]

  // VDTOR-THUNKS-Test5: Thunks for 'vdtor::Test5::~Test5()' (1 entry).
  // VDTOR-THUNKS-Test5-NEXT: 0 | [this adjustment: -4 non-virtual]

  // VDTOR-THUNKS-Test5: VFTable indices for 'vdtor::Test5' (1 entries).
  // VDTOR-THUNKS-Test5-NEXT: -- accessible via vfptr at offset 4 --
  // VDTOR-THUNKS-Test5-NEXT: 0 | vdtor::Test5::~Test5() [scalar deleting]
};

Test5 t5;

struct Test6 : Test4, Test2 {
  // Implicit virtual dtor here!

  // VDTOR-THUNKS-Test6: VFTable for 'vdtor::Test4' in 'vdtor::Test6' (1 entries).
  // VDTOR-THUNKS-Test6-NEXT: 0 | void vdtor::Test4::z4()

  // VDTOR-THUNKS-Test6: VFTable for 'vdtor::Test2' in 'vdtor::Test6' (1 entries).
  // VDTOR-THUNKS-Test6-NEXT: 0 | vdtor::Test6::~Test6() [scalar deleting]
  // VDTOR-THUNKS-Test6-NEXT:     [this adjustment: -4 non-virtual]

  // VDTOR-THUNKS-Test6: Thunks for 'vdtor::Test6::~Test6()' (1 entry).
  // VDTOR-THUNKS-Test6-NEXT: 0 | [this adjustment: -4 non-virtual]

  // VDTOR-THUNKS-Test6: VFTable indices for 'vdtor::Test6' (1 entries).
  // VDTOR-THUNKS-Test6-NEXT: -- accessible via vfptr at offset 4 --
  // VDTOR-THUNKS-Test6-NEXT: 0 | vdtor::Test6::~Test6() [scalar deleting]
};

Test6 t6;

struct Test7 : Test5 {
  // VDTOR-THUNKS-Test7: VFTable for 'vdtor::Test4' in 'vdtor::Test5' in 'vdtor::Test7' (1 entries).
  // VDTOR-THUNKS-Test7-NEXT: 0 | void vdtor::Test4::z4()

  // VDTOR-THUNKS-Test7: VFTable for 'vdtor::Test2' in 'vdtor::Test5' in 'vdtor::Test7' (1 entries).
  // VDTOR-THUNKS-Test7-NEXT: 0 | vdtor::Test7::~Test7() [scalar deleting]
  // VDTOR-THUNKS-Test7-NEXT:     [this adjustment: -4 non-virtual]

  // VDTOR-THUNKS-Test7: Thunks for 'vdtor::Test7::~Test7()' (1 entry).
  // VDTOR-THUNKS-Test7-NEXT: 0 | [this adjustment: -4 non-virtual]

  // VDTOR-THUNKS-Test7: VFTable indices for 'vdtor::Test7' (1 entries).
  // VDTOR-THUNKS-Test7-NEXT: -- accessible via vfptr at offset 4 --
  // VDTOR-THUNKS-Test7-NEXT: 0 | vdtor::Test7::~Test7() [scalar deleting]
  virtual ~Test7();
};

Test7 t7;

}

namespace return_adjustment {

struct Ret1 {
  virtual C* foo();
  virtual void z();
};

struct Test1 : Ret1 {
  // RET-THUNKS-Test1: VFTable for 'return_adjustment::Ret1' in 'return_adjustment::Test1' (3 entries).
  // RET-THUNKS-Test1-NEXT: 0 | this_adjustment::Test1 *return_adjustment::Test1::foo()
  // RET-THUNKS-Test1-NEXT:     [return adjustment: 4 non-virtual]
  // RET-THUNKS-Test1-NEXT: 1 | void return_adjustment::Ret1::z()
  // RET-THUNKS-Test1-NEXT: 2 | this_adjustment::Test1 *return_adjustment::Test1::foo()

  // RET-THUNKS-Test1: VFTable indices for 'return_adjustment::Test1' (1 entries).
  // RET-THUNKS-Test1-NEXT: 2 | this_adjustment::Test1 *return_adjustment::Test1::foo()

  // MANGLING-DAG: @"\01??_7Test1@return_adjustment@@6B@"

  virtual this_adjustment::Test1* foo();
};

Test1 t1;

struct Ret2 : B, this_adjustment::Test1 { };

struct Test2 : Test1 {
  // RET-THUNKS-Test2: VFTable for 'return_adjustment::Ret1' in 'return_adjustment::Test1' in 'return_adjustment::Test2' (4 entries).
  // RET-THUNKS-Test2-NEXT: 0 | return_adjustment::Ret2 *return_adjustment::Test2::foo()
  // RET-THUNKS-Test2-NEXT:     [return adjustment: 8 non-virtual]
  // RET-THUNKS-Test2-NEXT: 1 | void return_adjustment::Ret1::z()
  // RET-THUNKS-Test2-NEXT: 2 | return_adjustment::Ret2 *return_adjustment::Test2::foo()
  // RET-THUNKS-Test2-NEXT:     [return adjustment: 4 non-virtual]
  // RET-THUNKS-Test2-NEXT: 3 | return_adjustment::Ret2 *return_adjustment::Test2::foo()

  // RET-THUNKS-Test2: VFTable indices for 'return_adjustment::Test2' (1 entries).
  // RET-THUNKS-Test2-NEXT: 3 | return_adjustment::Ret2 *return_adjustment::Test2::foo()

  virtual Ret2* foo();
};

Test2 t2;

struct Test3: B, Ret1 {
  // RET-THUNKS-Test3: VFTable for 'B' in 'return_adjustment::Test3' (2 entries).
  // RET-THUNKS-Test3-NEXT: 0 | void B::g()
  // RET-THUNKS-Test3-NEXT: 1 | void B::h()

  // RET-THUNKS-Test3: VFTable for 'return_adjustment::Ret1' in 'return_adjustment::Test3' (3 entries).
  // RET-THUNKS-Test3-NEXT: 0 | this_adjustment::Test1 *return_adjustment::Test3::foo()
  // RET-THUNKS-Test3-NEXT:     [return adjustment: 4 non-virtual]
  // RET-THUNKS-Test3-NEXT: 1 | void return_adjustment::Ret1::z()
  // RET-THUNKS-Test3-NEXT: 2 | this_adjustment::Test1 *return_adjustment::Test3::foo()

  // RET-THUNKS-Test3: VFTable indices for 'return_adjustment::Test3' (1 entries).
  // RET-THUNKS-Test3-NEXT: via vfptr at offset 4
  // RET-THUNKS-Test3-NEXT: 2 | this_adjustment::Test1 *return_adjustment::Test3::foo()

  virtual this_adjustment::Test1* foo();
};

Test3 t3;

struct Test4 : Test3 {
  // RET-THUNKS-Test4: VFTable for 'B' in 'return_adjustment::Test3' in 'return_adjustment::Test4' (2 entries).
  // RET-THUNKS-Test4-NEXT: 0 | void B::g()
  // RET-THUNKS-Test4-NEXT: 1 | void B::h()

  // RET-THUNKS-Test4: VFTable for 'return_adjustment::Ret1' in 'return_adjustment::Test3' in 'return_adjustment::Test4' (4 entries).
  // RET-THUNKS-Test4-NEXT: 0 | return_adjustment::Ret2 *return_adjustment::Test4::foo()
  // RET-THUNKS-Test4-NEXT:     [return adjustment: 8 non-virtual]
  // RET-THUNKS-Test4-NEXT: 1 | void return_adjustment::Ret1::z()
  // RET-THUNKS-Test4-NEXT: 2 | return_adjustment::Ret2 *return_adjustment::Test4::foo()
  // RET-THUNKS-Test4-NEXT:     [return adjustment: 4 non-virtual]
  // RET-THUNKS-Test4-NEXT: 3 | return_adjustment::Ret2 *return_adjustment::Test4::foo()

  // RET-THUNKS-Test4: VFTable indices for 'return_adjustment::Test4' (1 entries).
  // RET-THUNKS-Test4-NEXT: -- accessible via vfptr at offset 4 --
  // RET-THUNKS-Test4-NEXT:   3 | return_adjustment::Ret2 *return_adjustment::Test4::foo()

  virtual Ret2* foo();
};

Test4 t4;

struct Test5 : Ret1, Test1 {
  // RET-THUNKS-Test5: VFTable for 'return_adjustment::Ret1' in 'return_adjustment::Test5' (3 entries).
  // RET-THUNKS-Test5-NEXT: 0 | return_adjustment::Ret2 *return_adjustment::Test5::foo()
  // RET-THUNKS-Test5-NEXT:     [return adjustment: 8 non-virtual]
  // RET-THUNKS-Test5-NEXT: 1 | void return_adjustment::Ret1::z()
  // RET-THUNKS-Test5-NEXT: 2 | return_adjustment::Ret2 *return_adjustment::Test5::foo()

  // RET-THUNKS-Test5: VFTable for 'return_adjustment::Ret1' in  'return_adjustment::Test1' in 'return_adjustment::Test5' (4 entries).
  // RET-THUNKS-Test5-NEXT: 0 | return_adjustment::Ret2 *return_adjustment::Test5::foo()
  // RET-THUNKS-Test5-NEXT:     [return adjustment: 8 non-virtual]
  // RET-THUNKS-Test5-NEXT:     [this adjustment: -4 non-virtual]
  // RET-THUNKS-Test5-NEXT: 1 | void return_adjustment::Ret1::z()
  // RET-THUNKS-Test5-NEXT: 2 | return_adjustment::Ret2 *return_adjustment::Test5::foo()
  // RET-THUNKS-Test5-NEXT:     [return adjustment: 4 non-virtual]
  // RET-THUNKS-Test5-NEXT:     [this adjustment: -4 non-virtual]
  // RET-THUNKS-Test5-NEXT: 3 | return_adjustment::Ret2 *return_adjustment::Test5::foo()
  // RET-THUNKS-Test5-NEXT:     [this adjustment: -4 non-virtual]

  // RET-THUNKS-Test5: VFTable indices for 'return_adjustment::Test5' (1 entries).
  // RET-THUNKS-Test5-NEXT: 2 | return_adjustment::Ret2 *return_adjustment::Test5::foo()

  virtual Ret2* foo();
};

Test5 t5;

struct Ret3 : this_adjustment::Test1 { };

struct Test6 : Test1 {
  virtual Ret3* foo();
  // RET-THUNKS-Test6: VFTable for 'return_adjustment::Ret1' in 'return_adjustment::Test1' in 'return_adjustment::Test6' (4 entries).
  // RET-THUNKS-Test6-NEXT: 0 | return_adjustment::Ret3 *return_adjustment::Test6::foo()
  // RET-THUNKS-Test6-NEXT:     [return adjustment: 4 non-virtual]
  // RET-THUNKS-Test6-NEXT: 1 | void return_adjustment::Ret1::z()
  // RET-THUNKS-Test6-NEXT: 2 | return_adjustment::Ret3 *return_adjustment::Test6::foo()
  // RET-THUNKS-Test6-NEXT: 3 | return_adjustment::Ret3 *return_adjustment::Test6::foo()

  // RET-THUNKS-Test6: VFTable indices for 'return_adjustment::Test6' (1 entries).
  // RET-THUNKS-Test6-NEXT: 3 | return_adjustment::Ret3 *return_adjustment::Test6::foo()
};

Test6 t6;

}
