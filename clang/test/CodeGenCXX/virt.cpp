// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -O0 -S %s -o %t-64.s &&
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s &&
// RUN: clang-cc -triple i386-apple-darwin -std=c++0x -O0 -S %s -o %t-32.s &&
// RUN: FileCheck -check-prefix LP32 -input-file=%t-32.s %s &&

// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -O3 -S %s -o %t-O3-64.s &&
// RUN: FileCheck -check-prefix LPOPT64 --input-file=%t-O3-64.s %s &&
// RUN: clang-cc -triple i386-apple-darwin -std=c++0x -O3 -S %s -o %t-O3-32.s &&
// RUN: FileCheck -check-prefix LPOPT32 -input-file=%t-O3-32.s %s &&

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
  // CHECK-LPOPT32: movl $4, __ZZ5test2vE2sz
  // CHECK-LPOPT64: movl $8, __ZZ5test2vE2sz(%rip)
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


struct test6_B2 { virtual void funcB2(); char b[1000]; };
struct test6_B1 : virtual test6_B2 { virtual void funcB1(); };

struct test6_D : test6_B2, virtual test6_B1 {
};

// CEHCK-LP32: .zerofill __DATA, __common, _d6, 2012, 4
// CHECK-LP64: .zerofill __DATA, __common, _d6, 2024, 4

struct test7_B2 { virtual void funcB2(); };
struct test7_B1 : virtual test7_B2 { virtual void funcB1(); };

struct test7_D : test7_B2, virtual test7_B1 {
};

// CEHCK-LP32: .zerofill __DATA, __common, _d7, 8, 3
// CHECK-LP64: .zerofill __DATA, __common, _d7, 16, 3


struct test3_B3 { virtual void funcB3(); };
struct test3_B2 : virtual test3_B3 { virtual void funcB2(); };
struct test3_B1 : virtual test3_B2 { virtual void funcB1(); };

struct test3_D  : virtual test3_B1 {
  virtual void funcD() { }
};

// CHECK-LP32:__ZTV7test3_D:
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .long __ZTI7test3_D
// CHECK-LP32: .long __ZN8test3_B36funcB3Ev
// CHECK-LP32: .long __ZN8test3_B26funcB2Ev
// CHECK-LP32: .long __ZN8test3_B16funcB1Ev
// CHECK-LP32: .long __ZN7test3_D5funcDEv

// CHECK-LP64:__ZTV7test3_D:
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .quad __ZTI7test3_D
// CHECK-LP64: .quad __ZN8test3_B36funcB3Ev
// CHECK-LP64: .quad __ZN8test3_B26funcB2Ev
// CHECK-LP64: .quad __ZN8test3_B16funcB1Ev
// CHECK-LP64: .quad __ZN7test3_D5funcDEv

struct test4_D : virtual B, virtual C {
};

// CHECK-LP32:__ZTV7test4_D:
// CHECK-LP32: .space 4
// CHECK-LP32: .long 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .long __ZTI7test4_D
// CHECK-LP32: .long __ZN1C4bee1Ev
// CHECK-LP32: .long __ZN1C4bee2Ev
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .long 4294967292
// CHECK-LP32: .long __ZTI7test4_D 
// CHECK-LP32: .long __ZN1B4bar1Ev
// CHECK-LP32: .long __ZN1B4bar2Ev

// CHECK-LP64:__ZTV7test4_D:
// CHECK-LP64: .space 8
// CHECK-LP64: .quad 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .quad __ZTI7test4_D
// CHECK-LP64: .quad __ZN1C4bee1Ev
// CHECK-LP64: .quad __ZN1C4bee2Ev
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .quad 18446744073709551608
// CHECK-LP64: .quad __ZTI7test4_D
// CHECK-LP64: .quad __ZN1B4bar1Ev
// CHECK-LP64: .quad __ZN1B4bar2Ev


struct test5_B3 { virtual void funcB3(); };
struct test5_B2 : virtual test5_B3 { virtual void funcB2(); };
struct test5_B1 : virtual test5_B2 { virtual void funcB1(); };

struct test5_B23 { virtual void funcB23(); };
struct test5_B22 : virtual test5_B23 { virtual void funcB22(); };
struct test5_B21 : virtual test5_B22 { virtual void funcB21(); };


struct B232 { virtual void funcB232(); };
struct B231 { virtual void funcB231(); };

struct test5_B33 { virtual void funcB33(); };
struct test5_B32 : virtual test5_B33, virtual B232 { virtual void funcB32(); };
struct test5_B31 : virtual test5_B32, virtual B231 { virtual void funcB31(); };

struct test5_D  : virtual test5_B1, virtual test5_B21, virtual test5_B31 {
  virtual void funcD() { }
};

// CHECK-LP32:__ZTV7test5_D:
// CHECK-LP32 .long 16
// CHECK-LP32 .long 12
// CHECK-LP32 .long 8
// CHECK-LP32 .long 8
// CHECK-LP32 .long 8
// CHECK-LP32 .long 4
// CHECK-LP32 .long 4
// CHECK-LP32 .long 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .space 4
// CHECK-LP32: .long __ZTI7test5_D
// CHECK-LP32: .long __ZN8test5_B36funcB3Ev
// CHECK-LP32: .long __ZN8test5_B26funcB2Ev
// CHECK-LP32: .long __ZN8test5_B16funcB1Ev
// CHECK-LP32: .long __ZN7test5_D5funcDEv
// CHECK-LP32 .space 4
// CHECK-LP32 .space 4
// CHECK-LP32 .space 4
// CHECK-LP32 .space 4
// CHECK-LP32: .space 4
// CHECK-LP32 .long -4
// CHECK-LP32: .long __ZTI7test5_D
// CHECK-LP32: .long __ZN9test5_B237funcB23Ev
// CHECK-LP32 .long __ZN9test5_B227funcB22Ev
// CHECK-LP32 .long __ZN9test5_B217funcB21Ev
// CHECK-LP32 .space 4
// CHECK-LP32 .long 8
// CHECK-LP32 .space 4
// CHECK-LP32 .space 4
// CHECK-LP32 .long 4
// CHECK-LP32 .space 4
// CHECK-LP32: .space 4
// CHECK-LP32 .long -8
// CHECK-LP32 .long __ZTI7test5_D
// CHECK-LP32 .long __ZN9test5_B337funcB33Ev
// CHECK-LP32 .long __ZN9test5_B327funcB32Ev
// CHECK-LP32: .long __ZN9test5_B317funcB31Ev
// CHECK-LP32: .space 4
// CHECK-LP32 .long -12
// CHECK-LP32: .long __ZTI7test5_D
// CHECK-LP32: .long __ZN4B2328funcB232Ev
// CHECK-LP32: .space 4
// CHECK-LP32 .long -16
// CHECK-LP32: .long __ZTI7test5_D
// CHECK-LP32: .long __ZN4B2318funcB231Ev

// CHECK-LP64:__ZTV7test5_D:
// CHECK-LP64 .quad 32
// CHECK-LP64 .quad 24
// CHECK-LP64 .quad 16
// CHECK-LP64 .quad 16
// CHECK-LP64 .quad 16
// CHECK-LP64 .quad 8
// CHECK-LP64 .quad 8
// CHECK-LP64 .quad 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .space 8
// CHECK-LP64: .quad __ZTI7test5_D
// CHECK-LP64: .quad __ZN8test5_B36funcB3Ev
// CHECK-LP64: .quad __ZN8test5_B26funcB2Ev
// CHECK-LP64: .quad __ZN8test5_B16funcB1Ev
// CHECK-LP64: .quad __ZN7test5_D5funcDEv
// CHECK-LP64 .space 8
// CHECK-LP64 .space 8
// CHECK-LP64 .space 8
// CHECK-LP64 .space 8
// CHECK-LP64: .space 8
// CHECK-LP64 .quad 18446744073709551608
// CHECK-LP64: .quad __ZTI7test5_D
// CHECK-LP64: .quad __ZN9test5_B237funcB23Ev
// CHECK-LP64 .quad __ZN9test5_B227funcB22Ev
// CHECK-LP64 .quad __ZN9test5_B217funcB21Ev
// CHECK-LP64 .space 8
// CHECK-LP64 .quad 16
// CHECK-LP64 .space 8
// CHECK-LP64 .space 8
// CHECK-LP64 .quad 8
// CHECK-LP64 .space 8
// CHECK-LP64: .space 8
// CHECK-LP64 .quad 18446744073709551600
// CHECK-LP64 .quad __ZTI7test5_D
// CHECK-LP64 .quad __ZN9test5_B337funcB33Ev
// CHECK-LP64 .quad __ZN9test5_B327funcB32Ev
// CHECK-LP64: .quad __ZN9test5_B317funcB31Ev
// CHECK-LP64: .space 8
// CHECK-LP64 .quad 18446744073709551592
// CHECK-LP64: .quad __ZTI7test5_D
// CHECK-LP64: .quad __ZN4B2328funcB232Ev
// CHECK-LP64: .space 8
// CHECK-LP64 .quad 18446744073709551584
// CHECK-LP64: .quad __ZTI7test5_D
// CHECK-LP64: .quad __ZN4B2318funcB231Ev




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
// CHECK-LP32: .long 8
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
// CHECK-LP64: .quad 16
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


test5_D d5;
test4_D d4;
test3_D d3;

test6_D d6;
test7_D d7;
