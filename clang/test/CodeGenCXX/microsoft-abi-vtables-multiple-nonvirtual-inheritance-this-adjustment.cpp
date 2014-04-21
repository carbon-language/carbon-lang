// RUN: %clang_cc1 %s -fno-rtti -triple=i386-pc-win32 -emit-llvm -o %t.ll -fdump-vtable-layouts >%t
// RUN: FileCheck %s < %t
// RUN: FileCheck --check-prefix=MANGLING %s < %t.ll

namespace test1 {
struct A {
  virtual void g();
  // Add an extra virtual method so it's easier to check for the absence of thunks.
  virtual void h();
};

struct B {
  virtual void g();  // Collides with A::g if both are bases of some class.
};

// Overrides methods of two bases at the same time, thus needing thunks.
struct X : A, B {
  // CHECK-LABEL: VFTable for 'test1::A' in 'test1::X' (2 entries).
  // CHECK-NEXT:   0 | void test1::X::g()
  // CHECK-NEXT:   1 | void test1::A::h()

  // CHECK-LABEL: VFTable for 'test1::B' in 'test1::X' (1 entry).
  // CHECK-NEXT:   0 | void test1::X::g()
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'void test1::X::g()' (1 entry).
  // CHECK-NEXT:   0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'test1::X' (1 entry).
  // CHECK-NEXT:   0 | void test1::X::g()

  // MANGLING-DAG: @"\01??_7X@test1@@6BA@1@@"
  // MANGLING-DAG: @"\01??_7X@test1@@6BB@1@@"

  virtual void g();
} x;

void build_vftable(X *obj) { obj->g(); }
}

namespace test2 {
struct A {
  virtual void f();
};

struct B {
  virtual void g();
  virtual void h();
};

struct C {
  virtual void g();
};

struct X : A, B, C {
  // CHECK-LABEL: VFTable for 'test2::A' in 'test2::X' (1 entry).
  // CHECK-NEXT:   0 | void test2::A::f()

  // CHECK-LABEL: VFTable for 'test2::B' in 'test2::X' (2 entries).
  // CHECK-NEXT:   0 | void test2::X::g()
  // CHECK-NEXT:   1 | void test2::B::h()

  // CHECK-LABEL: VFTable for 'test2::C' in 'test2::X' (1 entry).
  // CHECK-NEXT:   0 | void test2::X::g()
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'void test2::X::g()' (1 entry).
  // CHECK-NEXT:   0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'test2::X' (1 entry).
  // CHECK-NEXT:   via vfptr at offset 4
  // CHECK-NEXT:   0 | void test2::X::g()

  // MANGLING-DAG: @"\01??_7X@test2@@6BA@1@@"
  // MANGLING-DAG: @"\01??_7X@test2@@6BB@1@@"
  // MANGLING-DAG: @"\01??_7X@test2@@6BC@1@@"

  virtual void g();
} x;

void build_vftable(X *obj) { obj->g(); }
}

namespace test3 {
struct A {
  virtual void f();
};

struct B {
  virtual void g();
  virtual void h();
};

struct C: A, B {
  // Overrides only the left child's method (A::f), needs no thunks.
  virtual void f();
};

struct D: A, B {
  // Overrides only the right child's method (B::g),
  // needs this adjustment but not thunks.
  virtual void g();
};

// Overrides methods of two bases at the same time, thus needing thunks.
struct X: C, D {
  // CHECK-LABEL: VFTable for 'test3::A' in 'test3::C' in 'test3::X' (1 entry).
  // CHECK-NEXT:   0 | void test3::X::f()

  // CHECK-LABEL: VFTable for 'test3::B' in 'test3::C' in 'test3::X' (2 entries).
  // CHECK-NEXT:   0 | void test3::X::g()
  // CHECK-NEXT:   1 | void test3::B::h()

  // CHECK-LABEL: VFTable for 'test3::A' in 'test3::D' in 'test3::X' (1 entry).
  // CHECK-NEXT:   0 | void test3::X::f()
  // CHECK-NEXT:       [this adjustment: -8 non-virtual]

  // CHECK-LABEL: Thunks for 'void test3::X::f()' (1 entry).
  // CHECK-NEXT:   0 | [this adjustment: -8 non-virtual]

  // CHECK-LABEL: VFTable for 'test3::B' in 'test3::D' in 'test3::X' (2 entries).
  // CHECK-NEXT:   0 | void test3::X::g()
  // CHECK-NEXT:       [this adjustment: -8 non-virtual]
  // CHECK-NEXT:   1 | void test3::B::h()

  // CHECK-LABEL: Thunks for 'void test3::X::g()' (1 entry).
  // CHECK-NEXT:   0 | [this adjustment: -8 non-virtual]

  // CHECK-LABEL: VFTable indices for 'test3::X' (2 entries).
  // CHECK-NEXT:   via vfptr at offset 0
  // CHECK-NEXT:   0 | void test3::X::f()
  // CHECK-NEXT:   via vfptr at offset 4
  // CHECK-NEXT:   0 | void test3::X::g()

  virtual void f();
  virtual void g();
} x;

void build_vftable(X *obj) { obj->g(); }
}
