// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -O3 -S %s -o %t-64.s &&
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s &&
// RUN: clang-cc -triple i386-apple-darwin -std=c++0x -O3 -S %s -o %t-32.s &&
// RUN: FileCheck -check-prefix LP32 -input-file=%t-32.s %s &&
// RUN: true

struct B {
  virtual void bar1();
  virtual void bar2();
  int b;
};
void B::bar1() { }
void B::bar2() { }

struct C {
  virtual void bee1();
  virtual void bee2();
};
void C::bee1() { }
void C::bee2() { }

struct D {
  virtual void boo();
};
void D::boo() { }

struct D1 {
  virtual void bar();
  virtual void bar2();
  virtual void bar3();
  virtual void bar4();
  virtual void bar5();
  void *d1;
};
void D1::bar() { }

class F : virtual public D1, virtual public D {
public:
  virtual void foo();
  void *f;
};
void F::foo() { }

int j;
void test2() {
  F f;
  static int sz = (char *)(&f.f) - (char *)(&f);
  j = sz;
  // FIXME: These should result in a frontend constant a la fold, no run time
  // initializer
  // CHECK-LP32: movl $4, __ZZ5test2vE2sz
  // CHECK-LP64: movl $8, __ZZ5test2vE2sz(%rip)
}

static_assert(sizeof(F) == sizeof(void*)*4, "invalid vbase size");

struct E {
  int e;
};

static_assert (sizeof (C) == (sizeof(void *)), "vtable pointer layout");

class A : public E, public B, public C {
public:
  virtual void foo1();
  virtual void foo2();
  A() { }
  int a;
} *ap;
void A::foo1() { }
void A::foo2() { }

int main() {
  A a;
  B b;
  ap->e = 1;
  ap->b = 2;
}

// CHECK-LP32: main:
// CHECK-LP32: movl $1, 8(%eax)
// CHECK-LP32: movl $2, 4(%eax)

// CHECK-LP64: main:
// CHECK-LP64: movl $1, 12(%rax)
// CHECK-LP64: movl $2, 8(%rax)

// CHECK-LP64: __ZTV1B:
// CHECK-LP64: .space 8
// CHECK-LP64: .quad __ZTI1B
// CHECK-LP64: .quad __ZN1B4bar1Ev
// CHECK-LP64: .quad __ZN1B4bar2Ev

// CHECK-LP32: __ZTV1B:
// CHECK-LP32: .space 4
// CHECK-LP32: .long __ZTI1B
// CHECK-LP32: .long __ZN1B4bar1Ev
// CHECK-LP32: .long __ZN1B4bar2Ev

// CHECK-LP64: __ZTV1A:
// CHECK-LP64: .space 8
// CHECK-LP64: .quad __ZTI1A
// CHECK-LP64: .quad __ZN1B4bar1Ev
// CHECK-LP64: .quad __ZN1B4bar2Ev
// CHECK-LP64: .quad __ZN1A4foo1Ev
// CHECK-LP64: .quad __ZN1A4foo2Ev
// CHECK-LP64: .quad 18446744073709551600
// CHECK-LP64: .quad __ZTI1A
// CHECK-LP64: .quad __ZN1C4bee1Ev
// CHECK-LP64: .quad __ZN1C4bee2Ev

// CHECK-LP32: __ZTV1A:
// CHECK-LP32: .space 4
// CHECK-LP32: .long __ZTI1A
// CHECK-LP32: .long __ZN1B4bar1Ev
// CHECK-LP32: .long __ZN1B4bar2Ev
// CHECK-LP32: .long __ZN1A4foo1Ev
// CHECK-LP32: .long __ZN1A4foo2Ev
// CHECK-LP32: .long 4294967284
// CHECK-LP32: .long __ZTI1A
// CHECK-LP32: .long __ZN1C4bee1Ev
// CHECK-LP32: .long __ZN1C4bee2Ev

// CHECK-LP32:__ZTV1F:
// CHECK-LP32: .space 4
// CHECK-LP32 .long 8
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .long __ZTI1F
// CHECK-LP32: .long __ZN1D3booEv
// CHECK-LP32: .long __ZN1F3fooEv
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .long 4294967288
// CHECK-LP32: .long __ZTI1F
// CHECK-LP32: .long __ZN2D13barEv
// CHECK-LP32: .long __ZN2D14bar2Ev
// CHECK-LP32: .long __ZN2D14bar3Ev
// CHECK-LP32: .long __ZN2D14bar4Ev
// CHECK-LP32: .long __ZN2D14bar5Ev

// CHECK-LP64: __ZTV1F:
// CHECK-LP64: .space 8
// CHECK-LP64 .quad 16
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .quad __ZTI1F
// CHECK-LP64: .quad __ZN1D3booEv
// CHECK-LP64: .quad __ZN1F3fooEv
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .quad 18446744073709551600
// CHECK-LP64: .quad __ZTI1F
// CHECK-LP64: .quad __ZN2D13barEv
// CHECK-LP64: .quad __ZN2D14bar2Ev
// CHECK-LP64: .quad __ZN2D14bar3Ev
// CHECK-LP64: .quad __ZN2D14bar4Ev
// CHECK-LP64: .quad __ZN2D14bar5Ev
