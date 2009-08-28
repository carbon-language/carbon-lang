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

struct test12_A {
  virtual void foo0() { }
  virtual void foo();
} *test12_pa;

struct test12_B : public test12_A {
  virtual void foo() { }
} *test12_pb;

struct test12_D : public test12_B {
} *test12_pd;
void test12_foo() {
  test12_pa->foo0();
  test12_pb->foo0();
  test12_pd->foo0();
  test12_pa->foo();
  test12_pb->foo();
  test12_pd->foo();
  test12_pa->test12_A::foo();
}

// CHECK-LPOPT32:__Z10test12_foov:
// CHECK-LPOPT32: movl _test12_pa, %eax
// CHECK-LPOPT32-NEXT: movl (%eax), %ecx
// CHECK-LPOPT32-NEXT: movl %eax, (%esp)
// CHECK-LPOPT32-NEXT: call *(%ecx)
// CHECK-LPOPT32-NEXT: movl _test12_pb, %eax
// CHECK-LPOPT32-NEXT: movl (%eax), %ecx
// CHECK-LPOPT32-NEXT: movl %eax, (%esp)
// CHECK-LPOPT32-NEXT: call *(%ecx)
// CHECK-LPOPT32-NEXT: movl _test12_pd, %eax
// CHECK-LPOPT32-NEXT: movl (%eax), %ecx
// CHECK-LPOPT32-NEXT: movl %eax, (%esp)
// CHECK-LPOPT32-NEXT: call *(%ecx)
// CHECK-LPOPT32-NEXT: movl _test12_pa, %eax
// CHECK-LPOPT32-NEXT: movl (%eax), %ecx
// CHECK-LPOPT32-NEXT: movl %eax, (%esp)
// CHECK-LPOPT32-NEXT: call *4(%ecx)
// CHECK-LPOPT32-NEXT: movl _test12_pb, %eax
// CHECK-LPOPT32-NEXT: movl (%eax), %ecx
// CHECK-LPOPT32-NEXT: movl %eax, (%esp)
// CHECK-LPOPT32-NEXT: call *4(%ecx)
// CHECK-LPOPT32-NEXT: movl _test12_pd, %eax
// CHECK-LPOPT32-NEXT: movl (%eax), %ecx
// CHECK-LPOPT32-NEXT: movl %eax, (%esp)
// CHECK-LPOPT32-NEXT: call *4(%ecx)
// CHECK-LPOPT32-NEXT: movl _test12_pa, %eax
// CHECK-LPOPT32-NEXT: movl %eax, (%esp)
// CHECK-LPOPT32-NEXT: call L__ZN8test12_A3fooEv$stub

// CHECK-LPOPT64:__Z10test12_foov:
// CHECK-LPOPT64: movq _test12_pa(%rip), %rdi
// CHECK-LPOPT64-NEXT: movq (%rdi), %rax
// CHECK-LPOPT64-NEXT: call *(%rax)
// CHECK-LPOPT64-NEXT: movq _test12_pb(%rip), %rdi
// CHECK-LPOPT64-NEXT: movq (%rdi), %rax
// CHECK-LPOPT64-NEXT: call *(%rax)
// CHECK-LPOPT64-NEXT: movq _test12_pd(%rip), %rdi
// CHECK-LPOPT64-NEXT: movq (%rdi), %rax
// CHECK-LPOPT64-NEXT: call *(%rax)
// CHECK-LPOPT64-NEXT: movq _test12_pa(%rip), %rdi
// CHECK-LPOPT64-NEXT: movq (%rdi), %rax
// CHECK-LPOPT64-NEXT: call *8(%rax)
// CHECK-LPOPT64-NEXT: movq _test12_pb(%rip), %rdi
// CHECK-LPOPT64-NEXT: movq (%rdi), %rax
// CHECK-LPOPT64-NEXT: call *8(%rax)
// CHECK-LPOPT64-NEXT: movq _test12_pd(%rip), %rdi
// CHECK-LPOPT64-NEXT: movq (%rdi), %rax
// CHECK-LPOPT64-NEXT: call *8(%rax)
// CHECK-LPOPT64-NEXT: movq _test12_pa(%rip), %rdi
// CHECK-LPOPT64-NEXT: call __ZN8test12_A3fooEv

struct test6_B2 { virtual void funcB2(); char b[1000]; };
struct test6_B1 : virtual test6_B2 { virtual void funcB1(); };

struct test6_D : test6_B2, virtual test6_B1 {
};

// CHECK-LP32: .zerofill __DATA, __common, _d6, 2012, 4
// CHECK-LP64: .zerofill __DATA, __common, _d6, 2024, 4

struct test7_B2 { virtual void funcB2(); };
struct test7_B1 : virtual test7_B2 { virtual void funcB1(); };

struct test7_D : test7_B2, virtual test7_B1 {
};

// CHECK-LP32: .zerofill __DATA, __common, _d7, 8, 3
// CHECK-LP64: .zerofill __DATA, __common, _d7, 16, 3


struct test3_B3 { virtual void funcB3(); };
struct test3_B2 : virtual test3_B3 { virtual void funcB2(); };
struct test3_B1 : virtual test3_B2 { virtual void funcB1(); };

struct test3_D : virtual test3_B1 {
  virtual void funcD() { }
};

// CHECK-LP32:__ZTV7test3_D:
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long __ZTI7test3_D
// CHECK-LP32-NEXT: .long __ZN8test3_B36funcB3Ev
// CHECK-LP32-NEXT: .long __ZN8test3_B26funcB2Ev
// CHECK-LP32-NEXT: .long __ZN8test3_B16funcB1Ev
// CHECK-LP32-NEXT: .long __ZN7test3_D5funcDEv

// CHECK-LP64:__ZTV7test3_D:
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI7test3_D
// CHECK-LP64-NEXT: .quad __ZN8test3_B36funcB3Ev
// CHECK-LP64-NEXT: .quad __ZN8test3_B26funcB2Ev
// CHECK-LP64-NEXT: .quad __ZN8test3_B16funcB1Ev
// CHECK-LP64-NEXT: .quad __ZN7test3_D5funcDEv

struct test4_D : virtual B, virtual C {
};

// CHECK-LP32:__ZTV7test4_D:
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long __ZTI7test4_D
// CHECK-LP32-NEXT: .long __ZN1C4bee1Ev
// CHECK-LP32-NEXT: .long __ZN1C4bee2Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967292
// CHECK-LP32-NEXT: .long __ZTI7test4_D 
// CHECK-LP32-NEXT: .long __ZN1B4bar1Ev
// CHECK-LP32-NEXT: .long __ZN1B4bar2Ev

// CHECK-LP64:__ZTV7test4_D:
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI7test4_D
// CHECK-LP64-NEXT: .quad __ZN1C4bee1Ev
// CHECK-LP64-NEXT: .quad __ZN1C4bee2Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551608
// CHECK-LP64-NEXT: .quad __ZTI7test4_D
// CHECK-LP64-NEXT: .quad __ZN1B4bar1Ev
// CHECK-LP64-NEXT: .quad __ZN1B4bar2Ev


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
// CHECK-LP32-NEXT: .long 16
// CHECK-LP32-NEXT: .long 12
// CHECK-LP32-NEXT: .long 8
// CHECK-LP32-NEXT: .long 8
// CHECK-LP32-NEXT: .long 8
// CHECK-LP32-NEXT: .long 4
// CHECK-LP32-NEXT: .long 4
// CHECK-LP32-NEXT: .long 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long __ZTI7test5_D
// CHECK-LP32-NEXT: .long __ZN8test5_B36funcB3Ev
// CHECK-LP32-NEXT: .long __ZN8test5_B26funcB2Ev
// CHECK-LP32-NEXT: .long __ZN8test5_B16funcB1Ev
// CHECK-LP32-NEXT: .long __ZN7test5_D5funcDEv
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967292
// CHECK-LP32-NEXT: .long __ZTI7test5_D
// CHECK-LP32-NEXT: .long __ZN9test5_B237funcB23Ev
// CHECK-LP32-NEXT: .long __ZN9test5_B227funcB22Ev
// CHECK-LP32-NEXT: .long __ZN9test5_B217funcB21Ev
// CHECK-LP32 .space 4
// CHECK-LP32: .long 8
// CHECK-LP32 .space 4
// CHECK-LP32 .space 4				FIXME
// CHECK-LP32: .long 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967288
// CHECK-LP32-NEXT: .long __ZTI7test5_D
// CHECK-LP32-NEXT: .long __ZN9test5_B337funcB33Ev
// CHECK-LP32-NEXT: .long __ZN9test5_B327funcB32Ev
// CHECK-LP32-NEXT: .long __ZN9test5_B317funcB31Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967284
// CHECK-LP32-NEXT: .long __ZTI7test5_D
// CHECK-LP32-NEXT: .long __ZN4B2328funcB232Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967280
// CHECK-LP32-NEXT: .long __ZTI7test5_D
// CHECK-LP32-NEXT: .long __ZN4B2318funcB231Ev

// CHECK-LP64:__ZTV7test5_D:
// CHECK-LP64-NEXT: .quad 32
// CHECK-LP64-NEXT: .quad 24
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 8
// CHECK-LP64-NEXT: .quad 8
// CHECK-LP64-NEXT: .quad 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI7test5_D
// CHECK-LP64-NEXT: .quad __ZN8test5_B36funcB3Ev
// CHECK-LP64-NEXT: .quad __ZN8test5_B26funcB2Ev
// CHECK-LP64-NEXT: .quad __ZN8test5_B16funcB1Ev
// CHECK-LP64-NEXT: .quad __ZN7test5_D5funcDEv
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551608
// CHECK-LP64-NEXT: .quad __ZTI7test5_D
// CHECK-LP64-NEXT: .quad __ZN9test5_B237funcB23Ev
// CHECK-LP64-NEXT: .quad __ZN9test5_B227funcB22Ev
// CHECK-LP64-NEXT: .quad __ZN9test5_B217funcB21Ev
// CHECK-LP64 .space 8
// CHECK-LP64: .quad 16
// CHECK-LP64 .space 8
// CHECK-LP64 .space 8
// CHECK-LP64: .quad 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64: .quad 18446744073709551600
// CHECK-LP64-NEXT: .quad __ZTI7test5_D
// CHECK-LP64-NEXT: .quad __ZN9test5_B337funcB33Ev
// CHECK-LP64-NEXT: .quad __ZN9test5_B327funcB32Ev
// CHECK-LP64-NEXT: .quad __ZN9test5_B317funcB31Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551592
// CHECK-LP64-NEXT: .quad __ZTI7test5_D
// CHECK-LP64-NEXT: .quad __ZN4B2328funcB232Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551584
// CHECK-LP64-NEXT: .quad __ZTI7test5_D
// CHECK-LP64-NEXT: .quad __ZN4B2318funcB231Ev

struct test8_B1 {
  virtual void ftest8_B1() { }
};
struct test8_B2aa {
  virtual void ftest8_B2aa() { }
  int i;
};
struct test8_B2ab {
  virtual void ftest8_B2ab() { }
  int i;
};
struct test8_B2a : virtual test8_B2aa, virtual test8_B2ab {
  virtual void ftest8_B2a() { }
};
struct test8_B2b {
  virtual void ftest8_B2b() { }
};
struct test8_B2 : test8_B2a, test8_B2b {
  virtual void ftest8_B2() { }
};
struct test8_B3 {
  virtual void ftest8_B3() { }
};
class test8_D : test8_B1, test8_B2, test8_B3 {
};

// CHECK-LP32:__ZTV7test8_D:
// CHECK-LP32-NEXT: .long 24
// CHECK-LP32-NEXT: .long 16
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long __ZTI7test8_D
// CHECK-LP32-NEXT: .long __ZN8test8_B19ftest8_B1Ev
// CHECK-LP32-NEXT: .long 20
// CHECK-LP32-NEXT: .long 12
// CHECK-LP32-NEXT: .long 4294967292
// CHECK-LP32-NEXT: .long __ZTI7test8_D
// CHECK-LP32-NEXT: .long __ZN9test8_B2a10ftest8_B2aEv
// CHECK-LP32-NEXT: .long __ZN8test8_B29ftest8_B2Ev
// CHECK-LP32-NEXT: .long 4294967288
// CHECK-LP32-NEXT: .long __ZTI7test8_D
// CHECK-LP32-NEXT: .long __ZN9test8_B2b10ftest8_B2bEv
// CHECK-LP32-NEXT: .long 4294967284
// CHECK-LP32-NEXT: .long __ZTI7test8_D
// CHECK-LP32-NEXT: .long __ZN8test8_B39ftest8_B3Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967280
// CHECK-LP32-NEXT: .long __ZTI7test8_D
// CHECK-LP32-NEXT: .long __ZN10test8_B2aa11ftest8_B2aaEv
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967272
// CHECK-LP32-NEXT: .long __ZTI7test8_D
// CHECK-LP32-NEXT: .long __ZN10test8_B2ab11ftest8_B2abEv

// CHECK-LP64:__ZTV7test8_D:
// CHECK-LP64-NEXT: .quad 48
// CHECK-LP64-NEXT: .quad 32
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI7test8_D
// CHECK-LP64-NEXT: .quad __ZN8test8_B19ftest8_B1Ev
// CHECK-LP64-NEXT: .quad 40
// CHECK-LP64-NEXT: .quad 24
// CHECK-LP64-NEXT: .quad 18446744073709551608
// CHECK-LP64-NEXT: .quad __ZTI7test8_D
// CHECK-LP64-NEXT: .quad __ZN9test8_B2a10ftest8_B2aEv
// CHECK-LP64-NEXT: .quad __ZN8test8_B29ftest8_B2Ev
// CHECK-LP64-NEXT: .quad 18446744073709551600
// CHECK-LP64-NEXT: .quad __ZTI7test8_D
// CHECK-LP64-NEXT: .quad __ZN9test8_B2b10ftest8_B2bEv
// CHECK-LP64-NEXT: .quad 18446744073709551592
// CHECK-LP64-NEXT: .quad __ZTI7test8_D
// CHECK-LP64-NEXT: .quad __ZN8test8_B39ftest8_B3Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551584
// CHECK-LP64-NEXT: .quad __ZTI7test8_D
// CHECK-LP64-NEXT: .quad __ZN10test8_B2aa11ftest8_B2aaEv
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551568
// CHECK-LP64-NEXT: .quad __ZTI7test8_D
// CHECK-LP64-NEXT: .quad __ZN10test8_B2ab11ftest8_B2abEv


struct test9_B3 { virtual void funcB3(); int i; };
struct test9_B2 : virtual test9_B3 { virtual void funcB2(); int i; };
struct test9_B1 : virtual test9_B2 { virtual void funcB1(); int i; };

struct test9_B23 { virtual void funcB23(); int i; };
struct test9_B22 : virtual test9_B23 { virtual void funcB22(); int i; };
struct test9_B21 : virtual test9_B22 { virtual void funcB21(); int i; };


struct test9_B232 { virtual void funcB232(); int i; };
struct test9_B231 { virtual void funcB231(); int i; };

struct test9_B33 { virtual void funcB33(); int i; };
struct test9_B32 : virtual test9_B33, virtual test9_B232 { virtual void funcB32(); int i; };
struct test9_B31 : virtual test9_B32, virtual test9_B231 { virtual void funcB31(); int i; };

struct test9_D  : virtual test9_B1, virtual test9_B21, virtual test9_B31 {
  virtual void funcD() { }
};

// CHECK-LP64: __ZTV7test9_D:
// CHECK-LP64-NEXT: .quad 168
// CHECK-LP64-NEXT: .quad 152
// CHECK-LP64-NEXT: .quad 136
// CHECK-LP64-NEXT: .quad 120
// CHECK-LP64-NEXT: .quad 104
// CHECK-LP64-NEXT: .quad 88
// CHECK-LP64-NEXT: .quad 72
// CHECK-LP64-NEXT: .quad 56
// CHECK-LP64-NEXT: .quad 40
// CHECK-LP64-NEXT: .quad 24
// CHECK-LP64-NEXT: .quad 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI7test9_D
// CHECK-LP64-NEXT: .quad __ZN7test9_D5funcDEv
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 32
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 18446744073709551608
// CHECK-LP64-NEXT: .quad __ZTI7test9_D
// CHECK-LP64-NEXT: .quad __ZN8test9_B16funcB1Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 18446744073709551592
// CHECK-LP64-NEXT: .quad __ZTI7test9_D
// CHECK-LP64-NEXT: .quad __ZN8test9_B26funcB2Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551576
// CHECK-LP64-NEXT: .quad __ZTI7test9_D
// CHECK-LP64-NEXT: .quad __ZN8test9_B36funcB3Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 32
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 18446744073709551560
// CHECK-LP64-NEXT: .quad __ZTI7test9_D
// CHECK-LP64-NEXT: .quad __ZN9test9_B217funcB21Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 18446744073709551544
// CHECK-LP64-NEXT: .quad __ZTI7test9_D
// CHECK-LP64-NEXT: .quad __ZN9test9_B227funcB22Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551528
// CHECK-LP64-NEXT: .quad __ZTI7test9_D
// CHECK-LP64-NEXT: .quad __ZN9test9_B237funcB23Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 64
// CHECK-LP64-NEXT: .quad 48
// CHECK-LP64-NEXT: .quad 32
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 18446744073709551512
// CHECK-LP64-NEXT: .quad __ZTI7test9_D
// CHECK-LP64-NEXT: .quad __ZN9test9_B317funcB31Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 32
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 18446744073709551496
// CHECK-LP64-NEXT: .quad __ZTI7test9_D
// CHECK-LP64-NEXT: .quad __ZN9test9_B327funcB32Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551480
// CHECK-LP64-NEXT: .quad __ZTI7test9_D
// CHECK-LP64-NEXT: .quad __ZN9test9_B337funcB33Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551464
// CHECK-LP64-NEXT: .quad __ZTI7test9_D
// CHECK-LP64-NEXT: .quad __ZN10test9_B2328funcB232Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551448
// CHECK-LP64-NEXT: .quad __ZTI7test9_D
// CHECK-LP64-NEXT: .quad __ZN10test9_B2318funcB231Ev

// CHECK-LP32: __ZTV7test9_D:
// CHECK-LP32-NEXT: .long 84
// CHECK-LP32-NEXT: .long 76
// CHECK-LP32-NEXT: .long 68
// CHECK-LP32-NEXT: .long 60
// CHECK-LP32-NEXT: .long 52
// CHECK-LP32-NEXT: .long 44
// CHECK-LP32-NEXT: .long 36
// CHECK-LP32-NEXT: .long 28
// CHECK-LP32-NEXT: .long 20
// CHECK-LP32-NEXT: .long 12
// CHECK-LP32-NEXT: .long 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long __ZTI7test9_D
// CHECK-LP32-NEXT: .long __ZN7test9_D5funcDEv
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 16
// CHECK-LP32-NEXT: .long 8
// CHECK-LP32-NEXT: .long 4294967292
// CHECK-LP32-NEXT: .long __ZTI7test9_D
// CHECK-LP32-NEXT: .long __ZN8test9_B16funcB1Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 8
// CHECK-LP32-NEXT: .long 4294967284
// CHECK-LP32-NEXT: .long __ZTI7test9_D
// CHECK-LP32-NEXT: .long __ZN8test9_B26funcB2Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967276
// CHECK-LP32-NEXT: .long __ZTI7test9_D
// CHECK-LP32-NEXT: .long __ZN8test9_B36funcB3Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 16
// CHECK-LP32-NEXT: .long 8
// CHECK-LP32-NEXT: .long 4294967268
// CHECK-LP32-NEXT: .long __ZTI7test9_D
// CHECK-LP32-NEXT: .long __ZN9test9_B217funcB21Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 8
// CHECK-LP32-NEXT: .long 4294967260
// CHECK-LP32-NEXT: .long __ZTI7test9_D
// CHECK-LP32-NEXT: .long __ZN9test9_B227funcB22Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967252
// CHECK-LP32-NEXT: .long __ZTI7test9_D
// CHECK-LP32-NEXT: .long __ZN9test9_B237funcB23Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 32
// CHECK-LP32-NEXT: .long 24
// CHECK-LP32-NEXT: .long 16
// CHECK-LP32-NEXT: .long 8
// CHECK-LP32-NEXT: .long 4294967244
// CHECK-LP32-NEXT: .long __ZTI7test9_D
// CHECK-LP32-NEXT: .long __ZN9test9_B317funcB31Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 16
// CHECK-LP32-NEXT: .long 8
// CHECK-LP32-NEXT: .long 4294967236
// CHECK-LP32-NEXT: .long __ZTI7test9_D
// CHECK-LP32-NEXT: .long __ZN9test9_B327funcB32Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967228
// CHECK-LP32-NEXT: .long __ZTI7test9_D
// CHECK-LP32-NEXT: .long __ZN9test9_B337funcB33Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967220
// CHECK-LP32-NEXT: .long __ZTI7test9_D
// CHECK-LP32-NEXT: .long __ZN10test9_B2328funcB232Ev
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967212
// CHECK-LP32-NEXT: .long __ZTI7test9_D
// CHECK-LP32-NEXT: .long __ZN10test9_B2318funcB231Ev

struct test10_O { int i; };

struct test10_B1 : virtual test10_O {
  virtual void ftest10_B1() { }
};

struct test10_B2aa : virtual test10_O {
  int i;
};
struct test10_B2ab : virtual test10_O {
  int i;
};
struct test10_B2a : virtual test10_B2aa, virtual test10_B2ab,virtual test10_O {
  virtual void ftest10_B2a() { }
};
struct test10_B2b : virtual test10_O {
  virtual void ftest10_B2b() { }
};
struct test10_B2 : test10_B2a {
  virtual void ftest10_B2() { }
};
class test10_D : test10_B1, test10_B2 {
  
  void ftest10_B2aa() { }
};

// CHECK-LP64:__ZTV8test10_D:
// CHECK-LP64-NEXT: .quad 40
// CHECK-LP64-NEXT: .quad 24
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI8test10_D
// CHECK-LP64-NEXT: .quad __ZN9test10_B110ftest10_B1Ev
// CHECK-LP64-NEXT: .quad 32
// CHECK-LP64-NEXT: .quad 8
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 18446744073709551608
// CHECK-LP64-NEXT: .quad __ZTI8test10_D
// CHECK-LP64-NEXT: .quad __ZN10test10_B2a11ftest10_B2aEv
// CHECK-LP64-NEXT: .quad __ZN9test10_B210ftest10_B2Ev
// CHECK-LP64-NEXT: .quad 18446744073709551608
// CHECK-LP64-NEXT: .quad 18446744073709551592
// CHECK-LP64-NEXT: .quad __ZTI8test10_D
// CHECK-LP64-NEXT: .quad 18446744073709551592
// CHECK-LP64-NEXT: .quad 18446744073709551576
// CHECK-LP64-NEXT: .quad __ZTI8test10_D

// CHECK-LP32: __ZTV8test10_D:
// CHECK-LP32-NEXT: .long 20
// CHECK-LP32-NEXT: .long 12
// CHECK-LP32-NEXT: .long 8
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long __ZTI8test10_D
// CHECK-LP32-NEXT: .long __ZN9test10_B110ftest10_B1Ev
// CHECK-LP32-NEXT: .long 16
// CHECK-LP32-NEXT: .long 4
// CHECK-LP32-NEXT: .long 8
// CHECK-LP32-NEXT: .long 4294967292
// CHECK-LP32-NEXT: .long __ZTI8test10_D
// CHECK-LP32-NEXT: .long __ZN10test10_B2a11ftest10_B2aEv
// CHECK-LP32-NEXT: .long __ZN9test10_B210ftest10_B2Ev
// CHECK-LP32-NEXT: .long 4294967292
// CHECK-LP32-NEXT: .long 4294967284
// CHECK-LP32-NEXT: .long __ZTI8test10_D
// CHECK-LP32-NEXT: .long 4294967284
// CHECK-LP32-NEXT: .long 4294967276
// CHECK-LP32-NEXT: .long __ZTI8test10_D

struct test11_B {
  virtual void B1() { }
  virtual void D() { }
  virtual void B2() { }
};

struct test11_D : test11_B {
  virtual void D1() { }
  virtual void D() { }
  virtual void D2() { }
};

// CHECK-LP32:__ZTV8test11_D:
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long __ZTI8test11_D
// CHECK-LP32-NEXT: .long __ZN8test11_B2B1Ev
// CHECK-LP32-NEXT: .long __ZN8test11_D1DEv
// CHECK-LP32-NEXT: .long __ZN8test11_B2B2Ev
// CHECK-LP32-NEXT: .long __ZN8test11_D2D1Ev
// CHECK-LP32-NEXT: .long __ZN8test11_D2D2Ev


// CHECK-LP64:__ZTV8test11_D:
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI8test11_D
// CHECK-LP64-NEXT: .quad __ZN8test11_B2B1Ev
// CHECK-LP64-NEXT: .quad __ZN8test11_D1DEv
// CHECK-LP64-NEXT: .quad __ZN8test11_B2B2Ev
// CHECK-LP64-NEXT: .quad __ZN8test11_D2D1Ev
// CHECK-LP64-NEXT: .quad __ZN8test11_D2D2Ev


// CHECK-LP64: __ZTV1B:
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI1B
// CHECK-LP64-NEXT: .quad __ZN1B4bar1Ev
// CHECK-LP64-NEXT: .quad __ZN1B4bar2Ev

// CHECK-LP32: __ZTV1B:
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long __ZTI1B
// CHECK-LP32-NEXT: .long __ZN1B4bar1Ev
// CHECK-LP32-NEXT: .long __ZN1B4bar2Ev

// CHECK-LP64: __ZTV1A:
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI1A
// CHECK-LP64-NEXT: .quad __ZN1B4bar1Ev
// CHECK-LP64-NEXT: .quad __ZN1B4bar2Ev
// CHECK-LP64-NEXT: .quad __ZN1A4foo1Ev
// CHECK-LP64-NEXT: .quad __ZN1A4foo2Ev
// CHECK-LP64-NEXT: .quad 18446744073709551600
// CHECK-LP64-NEXT: .quad __ZTI1A
// CHECK-LP64-NEXT: .quad __ZN1C4bee1Ev
// CHECK-LP64-NEXT: .quad __ZN1C4bee2Ev

// CHECK-LP32: __ZTV1A:
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long __ZTI1A
// CHECK-LP32-NEXT: .long __ZN1B4bar1Ev
// CHECK-LP32-NEXT: .long __ZN1B4bar2Ev
// CHECK-LP32-NEXT: .long __ZN1A4foo1Ev
// CHECK-LP32-NEXT: .long __ZN1A4foo2Ev
// CHECK-LP32-NEXT: .long 4294967284
// CHECK-LP32-NEXT: .long __ZTI1A
// CHECK-LP32-NEXT: .long __ZN1C4bee1Ev
// CHECK-LP32-NEXT: .long __ZN1C4bee2Ev

// CHECK-LP32:__ZTV1F:
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 8
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long __ZTI1F
// CHECK-LP32-NEXT: .long __ZN1D3booEv
// CHECK-LP32-NEXT: .long __ZN1F3fooEv
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .space 4
// CHECK-LP32-NEXT: .long 4294967288
// CHECK-LP32-NEXT: .long __ZTI1F
// CHECK-LP32-NEXT: .long __ZN2D13barEv
// CHECK-LP32-NEXT: .long __ZN2D14bar2Ev
// CHECK-LP32-NEXT: .long __ZN2D14bar3Ev
// CHECK-LP32-NEXT: .long __ZN2D14bar4Ev
// CHECK-LP32-NEXT: .long __ZN2D14bar5Ev

// CHECK-LP64: __ZTV1F:
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI1F
// CHECK-LP64-NEXT: .quad __ZN1D3booEv
// CHECK-LP64-NEXT: .quad __ZN1F3fooEv
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551600
// CHECK-LP64-NEXT: .quad __ZTI1F
// CHECK-LP64-NEXT: .quad __ZN2D13barEv
// CHECK-LP64-NEXT: .quad __ZN2D14bar2Ev
// CHECK-LP64-NEXT: .quad __ZN2D14bar3Ev
// CHECK-LP64-NEXT: .quad __ZN2D14bar4Ev
// CHECK-LP64-NEXT: .quad __ZN2D14bar5Ev


test11_D d11;
test10_D d10;
test9_D d9;
test8_D d8;

test5_D d5;
test4_D d4;
test3_D d3;

test6_D d6;
test7_D d7;
