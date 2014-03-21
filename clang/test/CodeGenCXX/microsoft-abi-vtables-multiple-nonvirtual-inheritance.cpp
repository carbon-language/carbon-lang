// RUN: %clang_cc1 %s -fno-rtti -triple=i386-pc-win32 -emit-llvm -o %t.ll -fdump-vtable-layouts >%t
// RUN: FileCheck %s < %t
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
  // CHECK-LABEL:Test1' (1 entry)
  // CHECK-NEXT: 0 | void no_thunks::Test1::f()

  // CHECK-LABEL:Test1' (2 entries)
  // CHECK-NEXT: 0 | void B::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL:Test1' (1 entry)
  // CHECK-NEXT: 0 | void no_thunks::Test1::f()

  // MANGLING-DAG: @"\01??_7Test1@no_thunks@@6BA@@@"
  // MANGLING-DAG: @"\01??_7Test1@no_thunks@@6BB@@@"

  // Overrides only the left child's method (A::f), needs no thunks.
  virtual void f();
};

Test1 t1;
void use(Test1 *obj) { obj->f(); }

struct Test2: A, B {
  // CHECK-LABEL: VFTable for 'A' in 'no_thunks::Test2' (1 entry)
  // CHECK-NEXT: 0 | void A::f()

  // CHECK-LABEL: VFTable for 'B' in 'no_thunks::Test2' (2 entries)
  // CHECK-NEXT: 0 | void no_thunks::Test2::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: VFTable indices for 'no_thunks::Test2' (1 entry).
  // CHECK-NEXT: via vfptr at offset 4
  // CHECK-NEXT: 0 | void no_thunks::Test2::g()

  // Overrides only the right child's method (B::g), needs this adjustment but
  // not thunks.
  virtual void g();
};

Test2 t2;
void use(Test2 *obj) { obj->g(); }

struct Test3: A, B {
  // CHECK-LABEL: VFTable for 'A' in 'no_thunks::Test3' (2 entries)
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void no_thunks::Test3::i()

  // CHECK-LABEL: VFTable for 'B' in 'no_thunks::Test3' (2 entries)
  // CHECK-NEXT: 0 | void B::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: VFTable indices for 'no_thunks::Test3' (1 entry).
  // CHECK-NEXT: 1 | void no_thunks::Test3::i()

  // Only adds a new method.
  virtual void i();
};

Test3 t3;
void use(Test3 *obj) { obj->i(); }

// Only the right base has a vftable, so it's laid out before the left one!
struct Test4 : Empty, A {
  // CHECK-LABEL: VFTable for 'A' in 'no_thunks::Test4' (1 entry)
  // CHECK-NEXT: 0 | void no_thunks::Test4::f()

  // CHECK-LABEL: VFTable indices for 'no_thunks::Test4' (1 entry).
  // CHECK-NEXT: 0 | void no_thunks::Test4::f()

  // MANGLING-DAG: @"\01??_7Test4@no_thunks@@6B@"

  virtual void f();
};

Test4 t4;
void use(Test4 *obj) { obj->f(); }

// 2-level structure with repeating subobject types, but no thunks needed.
struct Test5: Test1, Test2 {
  // CHECK-LABEL: VFTable for 'A' in 'no_thunks::Test1' in 'no_thunks::Test5' (2 entries)
  // CHECK-NEXT: 0 | void no_thunks::Test1::f()
  // CHECK-NEXT: 1 | void no_thunks::Test5::z()

  // CHECK-LABEL: VFTable for 'B' in 'no_thunks::Test1' in 'no_thunks::Test5' (2 entries)
  // CHECK-NEXT: 0 | void B::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: VFTable for 'A' in 'no_thunks::Test2' in 'no_thunks::Test5' (1 entry)
  // CHECK-NEXT: 0 | void A::f()

  // CHECK-LABEL: VFTable for 'B' in 'no_thunks::Test2' in 'no_thunks::Test5' (2 entries)
  // CHECK-NEXT: 0 | void no_thunks::Test2::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: VFTable indices for 'no_thunks::Test5' (1 entry).
  // CHECK-NEXT: 1 | void no_thunks::Test5::z()

  // MANGLING-DAG: @"\01??_7Test5@no_thunks@@6BA@@Test1@1@@"
  // MANGLING-DAG: @"\01??_7Test5@no_thunks@@6BA@@Test2@1@@"
  // MANGLING-DAG: @"\01??_7Test5@no_thunks@@6BB@@Test1@1@@"
  // MANGLING-DAG: @"\01??_7Test5@no_thunks@@6BB@@Test2@1@@"

  virtual void z();
};

Test5 t5;
void use(Test5 *obj) { obj->z(); }

struct Test6: Test1 {
  // CHECK-LABEL: VFTable for 'A' in 'no_thunks::Test1' in 'no_thunks::Test6' (1 entry).
  // CHECK-NEXT: 0 | void no_thunks::Test6::f()

  // CHECK-LABEL: VFTable for 'B' in 'no_thunks::Test1' in 'no_thunks::Test6' (2 entries).
  // CHECK-NEXT: 0 | void B::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: VFTable indices for 'no_thunks::Test6' (1 entry).
  // CHECK-NEXT: 0 | void no_thunks::Test6::f()

  // MANGLING-DAG: @"\01??_7Test6@no_thunks@@6BA@@@"
  // MANGLING-DAG: @"\01??_7Test6@no_thunks@@6BB@@@"

  // Overrides both no_thunks::Test1::f and A::f.
  virtual void f();
};

Test6 t6;
void use(Test6 *obj) { obj->f(); }

struct Test7: Test2 {
  // CHECK-LABEL: VFTable for 'A' in 'no_thunks::Test2' in 'no_thunks::Test7' (1 entry).
  // CHECK-NEXT: 0 | void A::f()

  // CHECK-LABEL: VFTable for 'B' in 'no_thunks::Test2' in 'no_thunks::Test7' (2 entries).
  // CHECK-NEXT: 0 | void no_thunks::Test7::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: VFTable indices for 'no_thunks::Test7' (1 entry).
  // CHECK-NEXT: via vfptr at offset 4
  // CHECK-NEXT: 0 | void no_thunks::Test7::g()

  // Overrides both no_thunks::Test2::g and B::g.
  virtual void g();
};

Test7 t7;
void use(Test7 *obj) { obj->g(); }

struct Test8: Test3 {
  // CHECK-LABEL: VFTable for 'A' in 'no_thunks::Test3' in 'no_thunks::Test8' (2 entries).
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void no_thunks::Test3::i()

  // CHECK-LABEL: VFTable for 'B' in 'no_thunks::Test3' in 'no_thunks::Test8' (2 entries).
  // CHECK-NEXT: 0 | void no_thunks::Test8::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: VFTable indices for 'no_thunks::Test8' (1 entry).
  // CHECK-NEXT: via vfptr at offset 4
  // CHECK-NEXT: 0 | void no_thunks::Test8::g()

  // Overrides grandparent's B::g.
  virtual void g();
};

Test8 t8;
void use(Test8 *obj) { obj->g(); }

struct D : A {
  virtual void g();
};

// Repeating subobject.
struct Test9: A, D {
  // CHECK-LABEL: VFTable for 'A' in 'no_thunks::Test9' (2 entries).
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void no_thunks::Test9::h()

  // CHECK-LABEL: VFTable for 'A' in 'no_thunks::D' in 'no_thunks::Test9' (2 entries).
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void no_thunks::D::g()

  // CHECK-LABEL: VFTable indices for 'no_thunks::Test9' (1 entry).
  // CHECK-NEXT: 1 | void no_thunks::Test9::h()

  // MANGLING-DAG: @"\01??_7Test9@no_thunks@@6BA@@@"
  // MANGLING-DAG: @"\01??_7Test9@no_thunks@@6BD@1@@"

  virtual void h();
};

Test9 t9;
void use(Test9 *obj) { obj->h(); }
}

namespace pure_virtual {
struct D {
  virtual void g() = 0;
  virtual void h();
};


struct Test1: A, D {
  // CHECK: VFTable for 'A' in 'pure_virtual::Test1' (1 entry)
  // CHECK-NEXT: 0 | void A::f()

  // CHECK: VFTable for 'pure_virtual::D' in 'pure_virtual::Test1' (2 entries)
  // CHECK-NEXT: 0 | void pure_virtual::Test1::g()
  // CHECK-NEXT: 1 | void pure_virtual::D::h()

  // CHECK: VFTable indices for 'pure_virtual::Test1' (1 entry).
  // CHECK-NEXT: via vfptr at offset 4
  // CHECK-NEXT: 0 | void pure_virtual::Test1::g()

  // MANGLING-DAG: @"\01??_7Test1@pure_virtual@@6BA@@@"
  // MANGLING-DAG: @"\01??_7Test1@pure_virtual@@6BD@1@@"

  // Overrides only the right child's method (pure_virtual::D::g), needs this adjustment but
  // not thunks.
  virtual void g();
};

Test1 t1;
void use(Test1 *obj) { obj->g(); }
}

namespace this_adjustment {

// Overrides methods of two bases at the same time, thus needing thunks.
struct Test1 : B, C {
  // CHECK-LABEL: VFTable for 'B' in 'this_adjustment::Test1' (2 entries).
  // CHECK-NEXT: 0 | void this_adjustment::Test1::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: VFTable for 'C' in 'this_adjustment::Test1' (1 entry).
  // CHECK-NEXT: 0 | void this_adjustment::Test1::g()
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'void this_adjustment::Test1::g()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'this_adjustment::Test1' (1 entry).
  // CHECK-NEXT: 0 | void this_adjustment::Test1::g()

  // MANGLING-DAG: @"\01??_7Test1@this_adjustment@@6BB@@@"
  // MANGLING-DAG: @"\01??_7Test1@this_adjustment@@6BC@@@"

  virtual void g();
};

Test1 t1;
void use(Test1 *obj) { obj->g(); }

struct Test2 : A, B, C {
  // CHECK-LABEL: VFTable for 'A' in 'this_adjustment::Test2' (1 entry).
  // CHECK-NEXT: 0 | void A::f()

  // CHECK-LABEL: VFTable for 'B' in 'this_adjustment::Test2' (2 entries).
  // CHECK-NEXT: 0 | void this_adjustment::Test2::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: VFTable for 'C' in 'this_adjustment::Test2' (1 entry).
  // CHECK-NEXT: 0 | void this_adjustment::Test2::g()
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'void this_adjustment::Test2::g()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'this_adjustment::Test2' (1 entry).
  // CHECK-NEXT: via vfptr at offset 4
  // CHECK-NEXT: 0 | void this_adjustment::Test2::g()

  // MANGLING-DAG: @"\01??_7Test2@this_adjustment@@6BA@@@"
  // MANGLING-DAG: @"\01??_7Test2@this_adjustment@@6BB@@@"
  // MANGLING-DAG: @"\01??_7Test2@this_adjustment@@6BC@@@"

  virtual void g();
};

Test2 t2;
void use(Test2 *obj) { obj->g(); }

// Overrides methods of two bases at the same time, thus needing thunks.
struct Test3: no_thunks::Test1, no_thunks::Test2 {
  // CHECK-LABEL: VFTable for 'A' in 'no_thunks::Test1' in 'this_adjustment::Test3' (1 entry).
  // CHECK-NEXT: 0 | void this_adjustment::Test3::f()

  // CHECK-LABEL: VFTable for 'B' in 'no_thunks::Test1' in 'this_adjustment::Test3' (2 entries).
  // CHECK-NEXT: 0 | void this_adjustment::Test3::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: VFTable for 'A' in 'no_thunks::Test2' in 'this_adjustment::Test3' (1 entry).
  // CHECK-NEXT: 0 | void this_adjustment::Test3::f()
  // CHECK-NEXT:     [this adjustment: -8 non-virtual]

  // CHECK-LABEL: Thunks for 'void this_adjustment::Test3::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -8 non-virtual]

  // CHECK-LABEL: VFTable for 'B' in 'no_thunks::Test2' in 'this_adjustment::Test3' (2 entries).
  // CHECK-NEXT: 0 | void this_adjustment::Test3::g()
  // CHECK-NEXT:     [this adjustment: -8 non-virtual]
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: Thunks for 'void this_adjustment::Test3::g()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -8 non-virtual]

  // CHECK-LABEL: VFTable indices for 'this_adjustment::Test3' (2 entries).
  // CHECK-NEXT: via vfptr at offset 0
  // CHECK-NEXT: 0 | void this_adjustment::Test3::f()
  // CHECK-NEXT: via vfptr at offset 4
  // CHECK-NEXT: 0 | void this_adjustment::Test3::g()

  virtual void f();
  virtual void g();
};

Test3 t3;
void use(Test3 *obj) { obj->g(); }
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
  // CHECK-LABEL: VFTable for 'vdtor::Test1' in 'vdtor::Test3' (2 entries).
  // CHECK-NEXT: 0 | vdtor::Test3::~Test3() [scalar deleting]
  // CHECK-NEXT: 1 | void vdtor::Test1::z1()

  // CHECK-LABEL: VFTable for 'vdtor::Test2' in 'vdtor::Test3' (1 entry).
  // CHECK-NEXT: 0 | vdtor::Test3::~Test3() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'vdtor::Test3::~Test3()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'vdtor::Test3' (1 entry).
  // CHECK-NEXT: 0 | vdtor::Test3::~Test3() [scalar deleting]
  virtual ~Test3();
};

Test3 t3;
void use(Test3 *obj) { delete obj; }

struct Test4 {
  // No virtual destructor here!
  virtual void z4();
};

struct Test5 : Test4, Test2 {
  // Implicit virtual dtor here!

  // CHECK-LABEL: VFTable for 'vdtor::Test4' in 'vdtor::Test5' (1 entry).
  // CHECK-NEXT: 0 | void vdtor::Test4::z4()

  // CHECK-LABEL: VFTable for 'vdtor::Test2' in 'vdtor::Test5' (1 entry).
  // CHECK-NEXT: 0 | vdtor::Test5::~Test5() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'vdtor::Test5::~Test5()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'vdtor::Test5' (1 entry).
  // CHECK-NEXT: -- accessible via vfptr at offset 4 --
  // CHECK-NEXT: 0 | vdtor::Test5::~Test5() [scalar deleting]
};

Test5 t5;
void use(Test5 *obj) { delete obj; }

struct Test6 : Test4, Test2 {
  // Implicit virtual dtor here!

  // CHECK-LABEL: VFTable for 'vdtor::Test4' in 'vdtor::Test6' (1 entry).
  // CHECK-NEXT: 0 | void vdtor::Test4::z4()

  // CHECK-LABEL: VFTable for 'vdtor::Test2' in 'vdtor::Test6' (1 entry).
  // CHECK-NEXT: 0 | vdtor::Test6::~Test6() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'vdtor::Test6::~Test6()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'vdtor::Test6' (1 entry).
  // CHECK-NEXT: -- accessible via vfptr at offset 4 --
  // CHECK-NEXT: 0 | vdtor::Test6::~Test6() [scalar deleting]
};

Test6 t6;
void use(Test6 *obj) { delete obj; }

struct Test7 : Test5 {
  // CHECK-LABEL: VFTable for 'vdtor::Test4' in 'vdtor::Test5' in 'vdtor::Test7' (1 entry).
  // CHECK-NEXT: 0 | void vdtor::Test4::z4()

  // CHECK-LABEL: VFTable for 'vdtor::Test2' in 'vdtor::Test5' in 'vdtor::Test7' (1 entry).
  // CHECK-NEXT: 0 | vdtor::Test7::~Test7() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'vdtor::Test7::~Test7()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'vdtor::Test7' (1 entry).
  // CHECK-NEXT: -- accessible via vfptr at offset 4 --
  // CHECK-NEXT: 0 | vdtor::Test7::~Test7() [scalar deleting]
  virtual ~Test7();
};

Test7 t7;
void use(Test7 *obj) { delete obj; }

}

namespace return_adjustment {

struct Ret1 {
  virtual C* foo();
  virtual void z();
};

struct Test1 : Ret1 {
  // CHECK-LABEL: VFTable for 'return_adjustment::Ret1' in 'return_adjustment::Test1' (3 entries).
  // CHECK-NEXT: 0 | this_adjustment::Test1 *return_adjustment::Test1::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct C *'): 4 non-virtual]
  // CHECK-NEXT: 1 | void return_adjustment::Ret1::z()
  // CHECK-NEXT: 2 | this_adjustment::Test1 *return_adjustment::Test1::foo()

  // CHECK-LABEL: Thunks for 'this_adjustment::Test1 *return_adjustment::Test1::foo()' (1 entry).
  // CHECK-NEXT: 0 | [return adjustment (to type 'struct C *'): 4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'return_adjustment::Test1' (1 entry).
  // CHECK-NEXT: 2 | this_adjustment::Test1 *return_adjustment::Test1::foo()

  // MANGLING-DAG: @"\01??_7Test1@return_adjustment@@6B@"

  virtual this_adjustment::Test1* foo();
};

Test1 t1;
void use(Test1 *obj) { obj->foo(); }

struct Ret2 : B, this_adjustment::Test1 { };

struct Test2 : Test1 {
  // CHECK-LABEL: VFTable for 'return_adjustment::Ret1' in 'return_adjustment::Test1' in 'return_adjustment::Test2' (4 entries).
  // CHECK-NEXT: 0 | return_adjustment::Ret2 *return_adjustment::Test2::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct C *'): 8 non-virtual]
  // CHECK-NEXT: 1 | void return_adjustment::Ret1::z()
  // CHECK-NEXT: 2 | return_adjustment::Ret2 *return_adjustment::Test2::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct this_adjustment::Test1 *'): 4 non-virtual]
  // CHECK-NEXT: 3 | return_adjustment::Ret2 *return_adjustment::Test2::foo()

  // CHECK-LABEL: Thunks for 'return_adjustment::Ret2 *return_adjustment::Test2::foo()' (2 entries).
  // CHECK-NEXT: 0 | [return adjustment (to type 'struct this_adjustment::Test1 *'): 4 non-virtual]
  // CHECK-NEXT: 1 | [return adjustment (to type 'struct C *'): 8 non-virtual]

  // CHECK-LABEL: VFTable indices for 'return_adjustment::Test2' (1 entry).
  // CHECK-NEXT: 3 | return_adjustment::Ret2 *return_adjustment::Test2::foo()

  virtual Ret2* foo();
};

Test2 t2;
void use(Test2 *obj) { obj->foo(); }

struct Test3: B, Ret1 {
  // CHECK-LABEL: VFTable for 'B' in 'return_adjustment::Test3' (2 entries).
  // CHECK-NEXT: 0 | void B::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: VFTable for 'return_adjustment::Ret1' in 'return_adjustment::Test3' (3 entries).
  // CHECK-NEXT: 0 | this_adjustment::Test1 *return_adjustment::Test3::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct C *'): 4 non-virtual]
  // CHECK-NEXT: 1 | void return_adjustment::Ret1::z()
  // CHECK-NEXT: 2 | this_adjustment::Test1 *return_adjustment::Test3::foo()

  // CHECK-LABEL: Thunks for 'this_adjustment::Test1 *return_adjustment::Test3::foo()' (1 entry).
  // CHECK-NEXT: 0 | [return adjustment (to type 'struct C *'): 4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'return_adjustment::Test3' (1 entry).
  // CHECK-NEXT: via vfptr at offset 4
  // CHECK-NEXT: 2 | this_adjustment::Test1 *return_adjustment::Test3::foo()

  virtual this_adjustment::Test1* foo();
};

Test3 t3;
void use(Test3 *obj) { obj->foo(); }

struct Test4 : Test3 {
  // CHECK-LABEL: VFTable for 'B' in 'return_adjustment::Test3' in 'return_adjustment::Test4' (2 entries).
  // CHECK-NEXT: 0 | void B::g()
  // CHECK-NEXT: 1 | void B::h()

  // CHECK-LABEL: VFTable for 'return_adjustment::Ret1' in 'return_adjustment::Test3' in 'return_adjustment::Test4' (4 entries).
  // CHECK-NEXT: 0 | return_adjustment::Ret2 *return_adjustment::Test4::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct C *'): 8 non-virtual]
  // CHECK-NEXT: 1 | void return_adjustment::Ret1::z()
  // CHECK-NEXT: 2 | return_adjustment::Ret2 *return_adjustment::Test4::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct this_adjustment::Test1 *'): 4 non-virtual]
  // CHECK-NEXT: 3 | return_adjustment::Ret2 *return_adjustment::Test4::foo()

  // CHECK-LABEL: Thunks for 'return_adjustment::Ret2 *return_adjustment::Test4::foo()' (2 entries).
  // CHECK-NEXT: 0 | [return adjustment (to type 'struct this_adjustment::Test1 *'): 4 non-virtual]
  // CHECK-NEXT: 1 | [return adjustment (to type 'struct C *'): 8 non-virtual]

  // CHECK-LABEL: VFTable indices for 'return_adjustment::Test4' (1 entry).
  // CHECK-NEXT: -- accessible via vfptr at offset 4 --
  // CHECK-NEXT: 3 | return_adjustment::Ret2 *return_adjustment::Test4::foo()

  virtual Ret2* foo();
};

Test4 t4;
void use(Test4 *obj) { obj->foo(); }

struct Test5 : Ret1, Test1 {
  // CHECK-LABEL: VFTable for 'return_adjustment::Ret1' in 'return_adjustment::Test5' (3 entries).
  // CHECK-NEXT: 0 | return_adjustment::Ret2 *return_adjustment::Test5::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct C *'): 8 non-virtual]
  // CHECK-NEXT: 1 | void return_adjustment::Ret1::z()
  // CHECK-NEXT: 2 | return_adjustment::Ret2 *return_adjustment::Test5::foo()

  // CHECK-LABEL: Thunks for 'return_adjustment::Ret2 *return_adjustment::Test5::foo()' (1 entry).
  // CHECK-NEXT: 0 | [return adjustment (to type 'struct C *'): 8 non-virtual]

  // CHECK-LABEL: VFTable for 'return_adjustment::Ret1' in  'return_adjustment::Test1' in 'return_adjustment::Test5' (4 entries).
  // CHECK-NEXT: 0 | return_adjustment::Ret2 *return_adjustment::Test5::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct C *'): 8 non-virtual]
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]
  // CHECK-NEXT: 1 | void return_adjustment::Ret1::z()
  // CHECK-NEXT: 2 | return_adjustment::Ret2 *return_adjustment::Test5::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct this_adjustment::Test1 *'): 4 non-virtual]
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]
  // CHECK-NEXT: 3 | return_adjustment::Ret2 *return_adjustment::Test5::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct return_adjustment::Ret2 *'): 0 non-virtual]
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'return_adjustment::Ret2 *return_adjustment::Test5::foo()' (3 entries).
  // CHECK-NEXT: 0 | [return adjustment (to type 'struct return_adjustment::Ret2 *'): 0 non-virtual]
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]
  // CHECK-NEXT: 1 | [return adjustment (to type 'struct this_adjustment::Test1 *'): 4 non-virtual]
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]
  // CHECK-NEXT: 2 | [return adjustment (to type 'struct C *'): 8 non-virtual]
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'return_adjustment::Test5' (1 entry).
  // CHECK-NEXT: 2 | return_adjustment::Ret2 *return_adjustment::Test5::foo()

  virtual Ret2* foo();
};

Test5 t5;
void use(Test5 *obj) { obj->foo(); }

struct Ret3 : this_adjustment::Test1 { };

struct Test6 : Test1 {
  virtual Ret3* foo();
  // CHECK-LABEL: VFTable for 'return_adjustment::Ret1' in 'return_adjustment::Test1' in 'return_adjustment::Test6' (4 entries).
  // CHECK-NEXT: 0 | return_adjustment::Ret3 *return_adjustment::Test6::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct C *'): 4 non-virtual]
  // CHECK-NEXT: 1 | void return_adjustment::Ret1::z()
  // CHECK-NEXT: 2 | return_adjustment::Ret3 *return_adjustment::Test6::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct this_adjustment::Test1 *'): 0 non-virtual]
  // CHECK-NEXT: 3 | return_adjustment::Ret3 *return_adjustment::Test6::foo()

  // CHECK-LABEL: Thunks for 'return_adjustment::Ret3 *return_adjustment::Test6::foo()' (2 entries).
  // CHECK-NEXT: 0 | [return adjustment (to type 'struct this_adjustment::Test1 *'): 0 non-virtual]
  // CHECK-NEXT: 1 | [return adjustment (to type 'struct C *'): 4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'return_adjustment::Test6' (1 entry).
  // CHECK-NEXT: 3 | return_adjustment::Ret3 *return_adjustment::Test6::foo()
};

Test6 t6;
void use(Test6 *obj) { obj->foo(); }

}
