// RUN: %clang_cc1 %s -fno-rtti -cxx-abi microsoft -triple=i386-pc-win32 -emit-llvm -fdump-vtable-layouts -o %t.ll > %t
// RUN: FileCheck --check-prefix=EMITS-VFTABLE %s < %t.ll
// RUN: FileCheck --check-prefix=NO-VFTABLE %s < %t.ll
// RUN: FileCheck --check-prefix=CHECK-A %s < %t
// RUN: FileCheck --check-prefix=CHECK-B %s < %t
// RUN: FileCheck --check-prefix=CHECK-C %s < %t
// RUN: FileCheck --check-prefix=CHECK-D %s < %t
// RUN: FileCheck --check-prefix=CHECK-E %s < %t
// RUN: FileCheck --check-prefix=CHECK-F %s < %t
// RUN: FileCheck --check-prefix=CHECK-G %s < %t
// RUN: FileCheck --check-prefix=CHECK-I %s < %t
// RUN: FileCheck --check-prefix=CHECK-J %s < %t
// RUN: FileCheck --check-prefix=CHECK-K %s < %t
// RUN: FileCheck --check-prefix=CHECK-L %s < %t
// RUN: FileCheck --check-prefix=CHECK-M %s < %t
// RUN: FileCheck --check-prefix=CHECK-N %s < %t

struct A {
  // CHECK-A: VFTable for 'A' (3 entries)
  // CHECK-A-NEXT: 0 | void A::f()
  // CHECK-A-NEXT: 1 | void A::g()
  // CHECK-A-NEXT: 2 | void A::h()
  // CHECK-A: VFTable indices for 'A' (3 entries)
  // CHECK-A-NEXT: 0 | void A::f()
  // CHECK-A-NEXT: 1 | void A::g()
  // CHECK-A-NEXT: 2 | void A::h()

  virtual void f();
  virtual void g();
  virtual void h();
  int ia;
};
A a;
// EMITS-VFTABLE-DAG: @"\01??_7A@@6B@" = linkonce_odr unnamed_addr constant [3 x i8*]

struct B : A {
  // CHECK-B: VFTable for 'A' in 'B' (5 entries)
  // CHECK-B-NEXT: 0 | void B::f()
  // CHECK-B-NEXT: 1 | void A::g()
  // CHECK-B-NEXT: 2 | void A::h()
  // CHECK-B-NEXT: 3 | void B::i()
  // CHECK-B-NEXT: 4 | void B::j()
  // CHECK-B: VFTable indices for 'B' (3 entries)
  // CHECK-B-NEXT: 0 | void B::f()
  // CHECK-B-NEXT: 3 | void B::i()
  // CHECK-B-NEXT: 4 | void B::j()

  virtual void f();  // overrides A::f()
  virtual void i();
  virtual void j();
};
B b;
// EMITS-VFTABLE-DAG: @"\01??_7B@@6B@" = linkonce_odr unnamed_addr constant [5 x i8*]

struct C {
  // CHECK-C: VFTable for 'C' (2 entries)
  // CHECK-C-NEXT: 0 | C::~C() [scalar deleting]
  // CHECK-C-NEXT: 1 | void C::f()
  // CHECK-C: VFTable indices for 'C' (2 entries).
  // CHECK-C-NEXT: 0 | C::~C() [scalar deleting]
  // CHECK-C-NEXT: 1 | void C::f()

  virtual ~C();
  virtual void f();
};
void C::f() {}
// NO-VFTABLE-NOT: @"\01??_7C@@6B@"

struct D {
  // CHECK-D: VFTable for 'D' (2 entries)
  // CHECK-D-NEXT: 0 | void D::f()
  // CHECK-D-NEXT: 1 | D::~D() [scalar deleting]
  // CHECK-D: VFTable indices for 'D' (2 entries)
  // CHECK-D-NEXT: 0 | void D::f()
  // CHECK-D-NEXT: 1 | D::~D() [scalar deleting]

  virtual void f();
  virtual ~D();
};
D d;
// EMITS-VFTABLE-DAG: @"\01??_7D@@6B@" = linkonce_odr unnamed_addr constant [2 x i8*]

struct E : A {
  // CHECK-E: VFTable for 'A' in 'E' (5 entries)
  // CHECK-E-NEXT: 0 | void A::f()
  // CHECK-E-NEXT: 1 | void A::g()
  // CHECK-E-NEXT: 2 | void A::h()
  // CHECK-E-NEXT: 3 | E::~E() [scalar deleting]
  // CHECK-E-NEXT: 4 | void E::i()
  // CHECK-E: VFTable indices for 'E' (2 entries).
  // CHECK-E-NEXT: 3 | E::~E() [scalar deleting]
  // CHECK-E-NEXT: 4 | void E::i()

  // ~E would be the key method, but it isn't used, and MS ABI has no key
  // methods.
  virtual ~E();
  virtual void i();
};
void E::i() {}
// NO-VFTABLE-NOT: @"\01??_7E@@6B@"

struct F : A {
  // CHECK-F: VFTable for 'A' in 'F' (5 entries)
  // CHECK-F-NEXT: 0 | void A::f()
  // CHECK-F-NEXT: 1 | void A::g()
  // CHECK-F-NEXT: 2 | void A::h()
  // CHECK-F-NEXT: 3 | void F::i()
  // CHECK-F-NEXT: 4 | F::~F() [scalar deleting]
  // CHECK-F: VFTable indices for 'F' (2 entries).
  // CHECK-F-NEXT: 3 | void F::i()
  // CHECK-F-NEXT: 4 | F::~F() [scalar deleting]

  virtual void i();
  virtual ~F();
};
F f;
// EMITS-VFTABLE-DAG: @"\01??_7F@@6B@" = linkonce_odr unnamed_addr constant [5 x i8*]

struct G : E {
  // CHECK-G: VFTable for 'A' in 'E' in 'G' (6 entries)
  // CHECK-G-NEXT: 0 | void G::f()
  // CHECK-G-NEXT: 1 | void A::g()
  // CHECK-G-NEXT: 2 | void A::h()
  // CHECK-G-NEXT: 3 | G::~G() [scalar deleting]
  // CHECK-G-NEXT: 4 | void E::i()
  // CHECK-G-NEXT: 5 | void G::j()
  // CHECK-G: VFTable indices for 'G' (3 entries).
  // CHECK-G-NEXT: 0 | void G::f()
  // CHECK-G-NEXT: 3 | G::~G() [scalar deleting]
  // CHECK-G-NEXT: 5 | void G::j()

  virtual void f();  // overrides A::f()
  virtual ~G();
  virtual void j();
};
void G::j() {}
// NO-VFTABLE-NOT: @"\01??_7G@@6B@"

// Test that the usual Itanium-style key method does not emit a vtable.
struct H {
  virtual void f();
};
void H::f() {}
// NO-VFTABLE-NOT: @"\01??_7H@@6B@"

struct Empty { };

struct I : Empty {
  // CHECK-I: VFTable for 'I' (2 entries)
  // CHECK-I-NEXT: 0 | void I::f()
  // CHECK-I-NEXT: 1 | void I::g()
  virtual void f();
  virtual void g();
};

I i;

struct J {
  // CHECK-J: VFTable for 'J' (6 entries)
  // CHECK-J-NEXT: 0 | void J::foo(long)
  // CHECK-J-NEXT: 1 | void J::foo(int)
  // CHECK-J-NEXT: 2 | void J::foo(short)
  // CHECK-J-NEXT: 3 | void J::bar(long)
  // CHECK-J-NEXT: 4 | void J::bar(int)
  // CHECK-J-NEXT: 5 | void J::bar(short)
  virtual void foo(short);
  virtual void bar(short);
  virtual void foo(int);
  virtual void bar(int);
  virtual void foo(long);
  virtual void bar(long);
};

J j;

struct K : J {
  // CHECK-K: VFTable for 'J' in 'K' (9 entries)
  // CHECK-K-NEXT: 0 | void J::foo(long)
  // CHECK-K-NEXT: 1 | void J::foo(int)
  // CHECK-K-NEXT: 2 | void J::foo(short)
  // CHECK-K-NEXT: 3 | void J::bar(long)
  // CHECK-K-NEXT: 4 | void J::bar(int)
  // CHECK-K-NEXT: 5 | void J::bar(short)
  // CHECK-K-NEXT: 6 | void K::bar(double)
  // CHECK-K-NEXT: 7 | void K::bar(float)
  // CHECK-K-NEXT: 8 | void K::foo(float)
  virtual void bar(float);
  virtual void foo(float);
  virtual void bar(double);
};

K k;

struct L : J {
  // CHECK-L: VFTable for 'J' in 'L' (9 entries)
  // CHECK-L-NEXT: 0 | void J::foo(long)
  // CHECK-L-NEXT: 1 | void L::foo(int)
  // CHECK-L-NEXT: 2 | void J::foo(short)
  // CHECK-L-NEXT: 3 | void J::bar(long)
  // CHECK-L-NEXT: 4 | void J::bar(int)
  // CHECK-L-NEXT: 5 | void J::bar(short)
  // CHECK-L-NEXT: 6 | void L::foo(float)
  // CHECK-L-NEXT: 7 | void L::bar(double)
  // CHECK-L-NEXT: 8 | void L::bar(float)

  // This case is interesting. Since the J::foo(int) override is the first method in
  // the class, foo(float) precedes the bar(double) and bar(float) in the vftable.
  virtual void foo(int);
  virtual void bar(float);
  virtual void foo(float);
  virtual void bar(double);
};

L l;

struct M : J {
  // CHECK-M: VFTable for 'J' in 'M' (11 entries)
  // CHECK-M-NEXT:  0 | void J::foo(long)
  // CHECK-M-NEXT:  1 | void M::foo(int)
  // CHECK-M-NEXT:  2 | void J::foo(short)
  // CHECK-M-NEXT:  3 | void J::bar(long)
  // CHECK-M-NEXT:  4 | void J::bar(int)
  // CHECK-M-NEXT:  5 | void J::bar(short)
  // CHECK-M-NEXT:  6 | void M::foo(float)
  // CHECK-M-NEXT:  7 | void M::spam(long)
  // CHECK-M-NEXT:  8 | void M::spam(int)
  // CHECK-M-NEXT:  9 | void M::bar(double)
  // CHECK-M-NEXT: 10 | void M::bar(float)

  virtual void foo(int);
  virtual void spam(int);
  virtual void bar(float);
  virtual void bar(double);
  virtual void foo(float);
  virtual void spam(long);
};

M m;

struct N {
  // CHECK-N: VFTable for 'N' (4 entries)
  // CHECK-N-NEXT: 0 | void N::operator+(int)
  // CHECK-N-NEXT: 1 | void N::operator+(short)
  // CHECK-N-NEXT: 2 | void N::operator*(int)
  // CHECK-N-NEXT: 3 | void N::operator*(short)
  virtual void operator+(short);
  virtual void operator*(short);
  virtual void operator+(int);
  virtual void operator*(int);
};

N n;
