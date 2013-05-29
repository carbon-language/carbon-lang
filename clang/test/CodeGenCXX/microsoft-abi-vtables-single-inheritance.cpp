// RUN: %clang_cc1 %s -fno-rtti -cxx-abi microsoft -triple=i386-pc-win32 -emit-llvm -fdump-vtable-layouts -o - > %t 2>&1
// RUN: FileCheck --check-prefix=EMITS-VTABLE %s < %t
// RUN: FileCheck --check-prefix=NO-VTABLE %s < %t
// RUN: FileCheck --check-prefix=CHECK-A %s < %t
// RUN: FileCheck --check-prefix=CHECK-B %s < %t
// RUN: FileCheck --check-prefix=CHECK-C %s < %t
// RUN: FileCheck --check-prefix=CHECK-D %s < %t
// RUN: FileCheck --check-prefix=CHECK-E %s < %t
// RUN: FileCheck --check-prefix=CHECK-F %s < %t
// RUN: FileCheck --check-prefix=CHECK-G %s < %t

struct A {
  // CHECK-A: Vtable for 'A' (3 entries)
  // CHECK-A-NEXT: 0 | void A::f()
  // CHECK-A-NEXT: 1 | void A::g()
  // CHECK-A-NEXT: 2 | void A::h()
  virtual void f();
  virtual void g();
  virtual void h();
  int ia;
};
A a;
// EMITS-VTABLE-DAG: @"\01??_7A@@6B@" = linkonce_odr unnamed_addr constant [3 x i8*]

struct B : A {
  // CHECK-B: Vtable for 'B' (5 entries)
  // CHECK-B-NEXT: 0 | void B::f()
  // CHECK-B-NEXT: 1 | void A::g()
  // CHECK-B-NEXT: 2 | void A::h()
  // CHECK-B-NEXT: 3 | void B::i()
  // CHECK-B-NEXT: 4 | void B::j()
  virtual void f();  // overrides A::f()
  virtual void i();
  virtual void j();
};
B b;
// EMITS-VTABLE-DAG: @"\01??_7B@@6B@" = linkonce_odr unnamed_addr constant [5 x i8*]

struct C {
  // CHECK-C: Vtable for 'C' (2 entries)
  // CHECK-C-NEXT: 0 | C::~C() [scalar deleting]
  // CHECK-C-NEXT: 1 | void C::f()
  // CHECK-C: VTable indices for 'C' (2 entries).
  // CHECK-C-NEXT: 0 | C::~C() [scalar deleting]
  // CHECK-C-NEXT: 1 | void C::f()
  virtual ~C();

  virtual void f();
};
void C::f() {}
// NO-VTABLE-NOT: @"\01??_7C@@6B@"

struct D {
  // CHECK-D: Vtable for 'D' (2 entries)
  // CHECK-D-NEXT: 0 | void D::f()
  // CHECK-D-NEXT: 1 | D::~D() [scalar deleting]
  virtual void f();

  virtual ~D();
};
D d;
// EMITS-VTABLE-DAG: @"\01??_7D@@6B@" = linkonce_odr unnamed_addr constant [2 x i8*]

struct E : A {
  // CHECK-E: Vtable for 'E' (5 entries)
  // CHECK-E-NEXT: 0 | void A::f()
  // CHECK-E-NEXT: 1 | void A::g()
  // CHECK-E-NEXT: 2 | void A::h()
  // CHECK-E-NEXT: 3 | E::~E() [scalar deleting]
  // CHECK-E-NEXT: 4 | void E::i()
  // CHECK-E: VTable indices for 'E' (2 entries).
  // CHECK-E-NEXT: 3 | E::~E() [scalar deleting]
  // CHECK-E-NEXT: 4 | void E::i()

  // ~E would be the key method, but it isn't used, and MS ABI has no key
  // methods.
  virtual ~E();
  virtual void i();
};
void E::i() {}
// NO-VTABLE-NOT: @"\01??_7E@@6B@"

struct F : A {
  // CHECK-F: Vtable for 'F' (5 entries)
  // CHECK-F-NEXT: 0 | void A::f()
  // CHECK-F-NEXT: 1 | void A::g()
  // CHECK-F-NEXT: 2 | void A::h()
  // CHECK-F-NEXT: 3 | void F::i()
  // CHECK-F-NEXT: 4 | F::~F() [scalar deleting]
  // CHECK-F: VTable indices for 'F' (2 entries).
  // CHECK-F-NEXT: 3 | void F::i()
  // CHECK-F-NEXT: 4 | F::~F() [scalar deleting]
  virtual void i();
  virtual ~F();
};
F f;
// EMITS-VTABLE-DAG: @"\01??_7F@@6B@" = linkonce_odr unnamed_addr constant [5 x i8*]

struct G : E {
  // CHECK-G: Vtable for 'G' (6 entries)
  // CHECK-G-NEXT: 0 | void G::f()
  // CHECK-G-NEXT: 1 | void A::g()
  // CHECK-G-NEXT: 2 | void A::h()
  // CHECK-G-NEXT: 3 | G::~G() [scalar deleting]
  // CHECK-G-NEXT: 4 | void E::i()
  // CHECK-G-NEXT: 5 | void G::j()
  // CHECK-G: VTable indices for 'G' (3 entries).
  // CHECK-G-NEXT: 0 | void G::f()
  // CHECK-G-NEXT: 3 | G::~G() [scalar deleting]
  // CHECK-G-NEXT: 5 | void G::j()
  virtual void f();  // overrides A::f()
  virtual ~G();
  virtual void j();
};
void G::j() {}
// NO-VTABLE-NOT: @"\01??_7G@@6B@"

// Test that the usual Itanium-style key method does not emit a vtable.
struct H {
  virtual void f();
};
void H::f() {}
// NO-VTABLE-NOT: @"\01??_7H@@6B@"
