// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -O0 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s

// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -emit-llvm %s -o %t-64.ll
// RUN: FileCheck -check-prefix LPLL64 --input-file=%t-64.ll %s


// CHECK-LP64: main:
// CHECK-LP64: movl $1, 12(%rax)
// CHECK-LP64: movl $2, 8(%rax)

struct B {
  virtual void bar1();
  virtual void bar2();
  int b;
};
void B::bar1() { }
void B::bar2() { }

// CHECK-LP64: __ZTV1B:
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI1B
// CHECK-LP64-NEXT: .quad __ZN1B4bar1Ev
// CHECK-LP64-NEXT: .quad __ZN1B4bar2Ev

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


int j;
void *vp;
void test2() {
  F f;
  static int sz = (char *)(&f.f) - (char *)(&f);
  vp = &sz;
  j = sz;
  // FIXME: These should result in a frontend constant a la fold, no run time
  // initializer
  // CHECK-LPLL64: define void @_Z5test2v()
  // CHECK-LPLL64: = getelementptr inbounds %class.F* %f, i32 0, i32 1
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

// CHECK-LPLL64:define void @_Z10test12_foov() nounwind {
// CHECK-LPLL64:  call void %2(%class.test14* %tmp)
// CHECK-LPLL64:  call void %5(%class.test14* %tmp1)
// CHECK-LPLL64:  call void %8(%class.test14* %tmp3)
// CHECK-LPLL64:  call void %11(%class.test14* %tmp5)
// CHECK-LPLL64:  call void %14(%class.test14* %tmp7)
// CHECK-LPLL64:  call void %17(%class.test14* %tmp9)
// CHECK-LPLL64:  call void @_ZN8test12_A3fooEv(%class.test14* %tmp11)


struct test6_B2 { virtual void funcB2(); char b[1000]; };
struct test6_B1 : virtual test6_B2 { virtual void funcB1(); };

struct test6_D : test6_B2, virtual test6_B1 {
};

// CHECK-LP64: .zerofill __DATA, __common, _d6, 2024, 4

struct test7_B2 { virtual void funcB2(); };
struct test7_B1 : virtual test7_B2 { virtual void funcB1(); };

struct test7_D : test7_B2, virtual test7_B1 {
};

// CHECK-LP64: .zerofill __DATA, __common, _d7, 16, 3


struct test3_B3 { virtual void funcB3(); };
struct test3_B2 : virtual test3_B3 { virtual void funcB2(); };
struct test3_B1 : virtual test3_B2 { virtual void funcB1(); };

struct test3_D : virtual test3_B1 {
  virtual void funcD() { }
};

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
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551600
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

// CHECK-LP64:__ZTC7test8_D8_8test8_B2:
// CHECK-LP64-NEXT:        .quad   40
// CHECK-LP64-NEXT:        .quad   24
// CHECK-LP64-NEXT:        .space  8
// CHECK-LP64-NEXT:        .quad   __ZTI8test8_B2
// CHECK-LP64-NEXT:        .quad   __ZN9test8_B2a10ftest8_B2aEv
// CHECK-LP64-NEXT:        .quad   __ZN8test8_B29ftest8_B2Ev
// CHECK-LP64-NEXT:        .space  8
// CHECK-LP64-NEXT:        .quad   18446744073709551592
// CHECK-LP64-NEXT:        .quad   __ZTI8test8_B2
// CHECK-LP64-NEXT:        .quad   __ZN10test8_B2aa11ftest8_B2aaEv
// CHECK-LP64-NEXT:        .space  8
// CHECK-LP64-NEXT:        .quad   18446744073709551576
// CHECK-LP64-NEXT:        .quad   __ZTI8test8_B2
// CHECK-LP64-NEXT:        .quad   __ZN10test8_B2ab11ftest8_B2abEv

// CHECK-LP64:__ZTC7test8_D8_9test8_B2a:
// CHECK-LP64-NEXT: .quad   40
// CHECK-LP64-NEXT: .quad   24
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .quad   __ZTI9test8_B2a
// CHECK-LP64-NEXT: .quad   __ZN9test8_B2a10ftest8_B2aEv
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .quad   18446744073709551592
// CHECK-LP64-NEXT: .quad   __ZTI9test8_B2a
// CHECK-LP64-NEXT: .quad   __ZN10test8_B2aa11ftest8_B2aaEv
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .quad   18446744073709551576
// CHECK-LP64-NEXT: .quad   __ZTI9test8_B2a
// CHECK-LP64-NEXT: .quad   __ZN10test8_B2ab11ftest8_B2abEv

// CHECK-LP64:__ZTT7test8_D:
// CHECK-LP64-NEXT: .quad   (__ZTV7test8_D) + 32
// CHECK-LP64-NEXT: .quad   (__ZTC7test8_D8_8test8_B2) + 32
// CHECK-LP64-NEXT: .quad   (__ZTC7test8_D8_9test8_B2a) + 32
// CHECK-LP64-NEXT  .quad   (__ZTC7test8_D8_9test8_B2a) + 64
// CHECK-LP64-NEXT  .quad   (__ZTC7test8_D8_9test8_B2a) + 96
// CHECK-LP64-NEXT  .quad   (__ZTC7test8_D8_8test8_B2) + 72
// CHECK-LP64-NEXT  .quad   (__ZTC7test8_D8_8test8_B2) + 104
// CHECK-LP64-NEXT  .quad   (__ZTV7test8_D) + 72
// CHECK-LP64: .quad   (__ZTV7test8_D) + 160
// CHECK-LP64: .quad   (__ZTV7test8_D) + 192


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

// CHECK-LP64:__ZTV8test11_D:
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI8test11_D
// CHECK-LP64-NEXT: .quad __ZN8test11_B2B1Ev
// CHECK-LP64-NEXT: .quad __ZN8test11_D1DEv
// CHECK-LP64-NEXT: .quad __ZN8test11_B2B2Ev
// CHECK-LP64-NEXT: .quad __ZN8test11_D2D1Ev
// CHECK-LP64-NEXT: .quad __ZN8test11_D2D2Ev

struct test13_B {
  virtual void B1() { }
  virtual void D() { }
  virtual void Da();
  virtual void Db() { }
  virtual void Dc() { }
  virtual void B2() { }
  int i;
};


struct test13_NV1 {
  virtual void fooNV1() { }
  virtual void D() { }
};


struct test13_B2 : /* test13_NV1, */ virtual test13_B {
  virtual void B2a() { }
  virtual void B2() { }
  virtual void D() { }
  virtual void Da();
  virtual void Dd() { }
  virtual void B2b() { }
  int i;
};


struct test13_D : test13_NV1, virtual test13_B2 {
  virtual void D1() { }
  virtual void D() { }
  virtual void Db() { }
  virtual void Dd() { }
  virtual void D2() { }
  virtual void fooNV1() { }
};

// CHECK-LP64:__ZTV8test13_D:
// CHECK-LP64-NEXT: .quad 24
// CHECK-LP64-NEXT: .quad 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI8test13_D
// CHECK-LP64-NEXT: .quad __ZN8test13_D6fooNV1Ev
// CHECK-LP64-NEXT: .quad __ZN8test13_D1DEv
// CHECK-LP64-NEXT: .quad __ZN8test13_D2D1Ev
// CHECK-LP64-NEXT: .quad __ZN8test13_D2DbEv
// CHECK-LP64-NEXT: .quad __ZN8test13_D2DdEv
// CHECK-LP64-NEXT: .quad __ZN8test13_D2D2Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551608
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551608
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 18446744073709551608
// CHECK-LP64-NEXT: .quad __ZTI8test13_D
// CHECK-LP64-NEXT: .quad __ZN9test13_B23B2aEv
// CHECK-LP64-NEXT: .quad __ZN9test13_B22B2Ev
// CHECK-LP64-NEXT: .quad __ZTv0_n48_N8test13_D1DEv
// CHECK-LP64-NEXT: .quad __ZN9test13_B22DaEv
// CHECK-LP64-NEXT: .quad __ZTv0_n64_N8test13_D2DdEv
// CHECK-LP64-NEXT: .quad __ZN9test13_B23B2bEv
// CHECK-LP64-NEXT: .quad 18446744073709551600
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551592
// CHECK-LP64-NEXT: .quad 18446744073709551600
// CHECK-LP64-NEXT: .quad 18446744073709551592
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551592
// CHECK-LP64-NEXT: .quad __ZTI8test13_D
// CHECK-LP64-NEXT: .quad __ZN8test13_B2B1Ev
// CHECK-LP64-NEXT: .quad __ZTv0_n32_N8test13_D1DEv
// CHECK-LP64-NEXT: .quad __ZTv0_n40_N9test13_B22DaEv
// CHECK-LP64-NEXT: .quad __ZTv0_n48_N8test13_D2DbEv
// CHECK-LP64-NEXT: .quad __ZN8test13_B2DcEv
// CHECK-LP64-NEXT: .quad __ZTv0_n64_N9test13_B22B2Ev


class test14 {
public:
    virtual void initWithInt(int a);
    static test14 *withInt(int a);
};

void test14::initWithInt(int a) { }

test14 *test14::withInt(int a) {
  test14 *me = new test14;
  me->initWithInt(a);
  return me;
}


struct test15_B {
  virtual test15_B *foo1() { return 0; }
  virtual test15_B *foo2() { return 0; }
  virtual test15_B *foo3() { return 0; }
  int i;
};

struct test15_NV1 {
  virtual void fooNV1() { }
  int i;
};

struct test15_B2 : test15_NV1, virtual test15_B {
  virtual test15_B2 *foo1() { return 0; }
  virtual test15_B2 *foo2() { return 0; }
  int i;
};

struct test15_D : test15_NV1, virtual test15_B2 {
  virtual test15_D *foo1() { return 0; }
};

// CHECK-LP64:__ZTV8test15_D:
// CHECK-LP64-NEXT: .quad 32
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI8test15_D
// CHECK-LP64-NEXT: .quad __ZN10test15_NV16fooNV1Ev
// CHECK-LP64-NEXT: .quad __ZN8test15_D4foo1Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551600
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 18446744073709551600
// CHECK-LP64-NEXT: .quad __ZTI8test15_D
// CHECK-LP64-NEXT: .quad __ZN10test15_NV16fooNV1Ev
// CHECK-LP64-NEXT: .quad __ZTcv0_n40_v0_n24_N8test15_D4foo1Ev
// CHECK-LP64-NEXT: .quad __ZN9test15_B24foo2Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551600
// CHECK-LP64-NEXT: .quad 18446744073709551584
// CHECK-LP64-NEXT: .quad 18446744073709551584
// CHECK-LP64-NEXT: .quad __ZTI8test15_D
// CHECK-LP64-NEXT: .quad __ZTcv0_n24_v0_n32_N8test15_D4foo1Ev
// CHECK-LP64-NEXT: .quad __ZTcv0_n32_v0_n24_N9test15_B24foo2Ev
// CHECK-LP64-NEXT: .quad __ZN8test15_B4foo3Ev


struct test16_NV1 {
  virtual void fooNV1() { }
virtual void foo_NV1() { }
  int i;
};

struct test16_NV2 {
  virtual test16_NV2* foo1() { return 0; }
virtual void foo_NV2() { }
virtual void foo_NV2b() { }
  int i;
};

struct test16_B : public test16_NV1, test16_NV2 {
  virtual test16_B *foo1() { return 0; }
  virtual test16_B *foo2() { return 0; }
  virtual test16_B *foo3() { return 0; }
virtual void foo_B() { }
  int i;
};

struct test16_B2 : test16_NV1, virtual test16_B {
  virtual test16_B2 *foo1() { return 0; }
  virtual test16_B2 *foo2() { return 0; }
virtual void foo_B2() { }
  int i;
};

struct test16_D : test16_NV1, virtual test16_B2 {
  virtual void bar() { }
  virtual test16_D *foo1() { return 0; }
};

// CHECK-LP64: __ZTV8test16_D:
// CHECK-LP64-NEXT: .quad 32
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI8test16_D
// CHECK-LP64-NEXT: .quad __ZN10test16_NV16fooNV1Ev
// CHECK-LP64-NEXT: .quad __ZN10test16_NV17foo_NV1Ev
// CHECK-LP64-NEXT: .quad __ZN8test16_D3barEv
// CHECK-LP64-NEXT: .quad __ZN8test16_D4foo1Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551600
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 18446744073709551600
// CHECK-LP64-NEXT: .quad __ZTI8test16_D
// CHECK-LP64-NEXT: .quad __ZN10test16_NV16fooNV1Ev
// CHECK-LP64-NEXT: .quad __ZN10test16_NV17foo_NV1Ev
// CHECK-LP64-NEXT: .quad __ZTcv0_n48_v0_n24_N8test16_D4foo1Ev
// CHECK-LP64-NEXT: .quad __ZN9test16_B24foo2Ev
// CHECK-LP64-NEXT: .quad __ZN9test16_B26foo_B2Ev
// CHECK-LP64-NEXT .quad 16
// CHECK-LP64-NEXT .quad 16
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64: .quad 18446744073709551600
// CHECK-LP64-NEXT: .quad 18446744073709551584
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551584
// CHECK-LP64-NEXT: .quad __ZTI8test16_D
// CHECK-LP64-NEXT: .quad __ZN10test16_NV16fooNV1Ev
// CHECK-LP64-NEXT: .quad __ZN10test16_NV17foo_NV1Ev
// CHECK-LP64-NEXT: .quad __ZTcv0_n40_v0_n32_N8test16_D4foo1Ev
// CHECK-LP64-NEXT: .quad __ZTcv0_n48_v0_n24_N9test16_B24foo2Ev
// CHECK-LP64-NEXT: .quad __ZN8test16_B4foo3Ev
// CHECK-LP64-NEXT: .quad __ZN8test16_B5foo_BEv
// CHECK-LP64-NEXT: .quad 18446744073709551568
// CHECK-LP64-NEXT: .quad __ZTI8test16_D
// CHECK-LP64-NEXT .quad __ZTcvn16_n40_v16_n32_N8test16_D4foo1Ev
// CHECK-LP64: .quad __ZN10test16_NV27foo_NV2Ev
// CHECK-LP64-NEXT: .quad __ZN10test16_NV28foo_NV2bEv


// FIXME: This is the wrong thunk, but until these issues are fixed, better
// than nothing.
// CHECK-LPLL64:define weak %class.test8_D* @_ZTcvn16_n72_v16_n32_N8test16_D4foo1Ev(%class.test8_D*)
// CHECK-LPLL64:entry:
// CHECK-LPLL64:  %retval = alloca %class.test8_D*
// CHECK-LPLL64:  %.addr = alloca %class.test8_D*
// CHECK-LPLL64:  store %class.test8_D* %0, %class.test8_D** %.addr
// CHECK-LPLL64:  %this = load %class.test8_D** %.addr
// CHECK-LPLL64:  %1 = bitcast %class.test8_D* %this to i8*
// CHECK-LPLL64:  %2 = getelementptr inbounds i8* %1, i64 -16
// CHECK-LPLL64:  %3 = bitcast i8* %2 to %class.test8_D*
// CHECK-LPLL64:  %4 = bitcast %class.test8_D* %3 to i8*
// CHECK-LPLL64:  %5 = bitcast %class.test8_D* %3 to i64**
// CHECK-LPLL64:  %vtable = load i64** %5
// CHECK-LPLL64:  %6 = getelementptr inbounds i64* %vtable, i64 -9
// CHECK-LPLL64:  %7 = load i64* %6
// CHECK-LPLL64:  %8 = getelementptr i8* %4, i64 %7
// CHECK-LPLL64:  %9 = bitcast i8* %8 to %class.test8_D*
// CHECK-LPLL64:  %call = call %class.test8_D* @_ZTch0_v16_n32_N8test16_D4foo1Ev(%class.test8_D* %9)
// CHECK-LPLL64:  store %class.test8_D* %call, %class.test8_D** %retval
// CHECK-LPLL64:  %10 = load %class.test8_D** %retval
// CHECK-LPLL64:  ret %class.test8_D* %10
// CHECK-LPLL64:}

// CHECK-LPLL64:define weak %class.test8_D* @_ZTch0_v16_n32_N8test16_D4foo1Ev(%class.test8_D*)
// CHECK-LPLL64:entry:
// CHECK-LPLL64:  %retval = alloca %class.test8_D*
// CHECK-LPLL64:  %.addr = alloca %class.test8_D*
// CHECK-LPLL64:  store %class.test8_D* %0, %class.test8_D** %.addr
// CHECK-LPLL64:  %this = load %class.test8_D** %.addr
// CHECK-LPLL64:  %call = call %class.test8_D* @_ZN8test16_D4foo1Ev(%class.test8_D* %this)
// CHECK-LPLL64:  %1 = icmp ne %class.test8_D* %call, null
// CHECK-LPLL64:  br i1 %1, label %2, label %12
// CHECK-LPLL64:; <label>:2
// CHECK-LPLL64:  %3 = bitcast %class.test8_D* %call to i8*
// CHECK-LPLL64:  %4 = getelementptr inbounds i8* %3, i64 16
// CHECK-LPLL64:  %5 = bitcast i8* %4 to %class.test8_D*
// CHECK-LPLL64:  %6 = bitcast %class.test8_D* %5 to i8*
// CHECK-LPLL64:  %7 = bitcast %class.test8_D* %5 to i64**
// CHECK-LPLL64:  %vtable = load i64** %7
// CHECK-LPLL64:  %8 = getelementptr inbounds i64* %vtable, i64 -4
// CHECK-LPLL64:  %9 = load i64* %8
// CHECK-LPLL64:  %10 = getelementptr i8* %6, i64 %9
// CHECK-LPLL64:  %11 = bitcast i8* %10 to %class.test8_D*
// CHECK-LPLL64:  br label %13
// CHECK-LPLL64:; <label>:12
// CHECK-LPLL64:  br label %13
// CHECK-LPLL64:; <label>:13
// CHECK-LPLL64:  %14 = phi %class.test8_D* [ %11, %2 ], [ %call, %12 ]
// CHECK-LPLL64:  store %class.test8_D* %14, %class.test8_D** %retval
// CHECK-LPLL64:  %15 = load %class.test8_D** %retval
// CHECK-LPLL64:  ret %class.test8_D* %15
// CHECK-LPLL64:}


class test17_B1 {
  virtual void foo() = 0;
  virtual void bar() { }
};

class test17_B2 : public test17_B1 {
  void foo() { }
  virtual void bar() = 0;
};

class test17_D : public test17_B2 {
  void bar() { }
};


// CHECK-LP64:__ZTV8test17_D:
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI8test17_D
// CHECK-LP64-NEXT: .quad __ZN9test17_B23fooEv
// CHECK-LP64-NEXT: .quad __ZN8test17_D3barEv

// CHECK-LP64:__ZTV9test17_B2:
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI9test17_B2
// CHECK-LP64-NEXT: .quad __ZN9test17_B23fooEv
// CHECK-LP64-NEXT: .quad ___cxa_pure_virtual

// CHECK-LP64:__ZTV9test17_B1:
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI9test17_B1
// CHECK-LP64-NEXT: .quad ___cxa_pure_virtual
// CHECK-LP64-NEXT: .quad __ZN9test17_B13barEv


struct test18_NV1 {
  virtual void fooNV1() { }
virtual void foo_NV1() { }
  int i;
};

struct test18_NV2 {
  virtual test18_NV2& foo1() { return *this; }
virtual void foo_NV2() { }
virtual void foo_NV2b() { }
  int i;
};

struct test18_B : public test18_NV1, test18_NV2 {
  virtual test18_B& foo1() { return *this; }
  virtual test18_B *foo2() { return 0; }
  virtual test18_B *foo3() { return 0; }
virtual void foo_B() { }
  int i;
};

struct test18_B2 : test18_NV1, virtual test18_B {
  virtual test18_B2& foo1() { return *this; }
  virtual test18_B2 *foo2() { return 0; }
virtual void foo_B2() { }
  int i;
};

struct test18_D : test18_NV1, virtual test18_B2 {
  virtual test18_D& foo1() { return *this; }
};


struct test19_VB1 { };
struct test19_B1 : public virtual test19_VB1 {
  virtual void fB1() { }
  virtual void foB1B2() { }
  virtual void foB1B3() { }
  virtual void foB1B4() { }
};

struct test19_VB2 { };
struct test19_B2: public test19_B1, public virtual test19_VB2 {
  virtual void foB1B2() { }
  virtual void foB1B3() { }
  virtual void foB1B4() { }

  virtual void fB2() { }
  virtual void foB2B3() { }
  virtual void foB2B4() { }
};

struct test19_VB3 { };
struct test19_B3: virtual public test19_B2, public virtual test19_VB3 {
  virtual void foB1B3() { }
  virtual void foB1B4() { }

  virtual void foB2B3() { }
  virtual void foB2B4() { }

  virtual void fB3() { }
  virtual void foB3B4() { }
};

struct test19_VB4 { };
struct test19_B4: public test19_B3, public virtual test19_VB4 {
  virtual void foB1B4() { }

  virtual void foB2B4() { }

  virtual void foB3B4() { }

  virtual void fB4() { }
};

struct test19_D : virtual test19_B4 {
};


// CHECK-LP64: __ZTV8test19_D:
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI8test19_D
// CHECK-LP64-NEXT: .quad __ZN9test19_B13fB1Ev
// CHECK-LP64-NEXT: .quad __ZN9test19_B26foB1B2Ev
// CHECK-LP64-NEXT: .quad __ZN9test19_B36foB1B3Ev
// CHECK-LP64-NEXT: .quad __ZN9test19_B46foB1B4Ev
// CHECK-LP64-NEXT: .quad __ZN9test19_B23fB2Ev
// CHECK-LP64-NEXT: .quad __ZN9test19_B36foB2B3Ev
// CHECK-LP64-NEXT: .quad __ZN9test19_B46foB2B4Ev
// CHECK-LP64-NEXT: .quad __ZN9test19_B33fB3Ev
// CHECK-LP64-NEXT: .quad __ZN9test19_B46foB3B4Ev
// CHECK-LP64-NEXT: .quad __ZN9test19_B43fB4Ev

// CHECK-LP64:     __ZTT8test19_D:
// CHECK-LP64-NEXT: .quad   (__ZTV8test19_D) + 144
// CHECK-LP64-NEXT: .quad   (__ZTV8test19_D) + 144
// CHECK-LP64-NEXT .quad   (__ZTV8test19_D) + 144
// CHECK-LP64-NEXT .quad   (__ZTC8test19_D0_9test19_B4) + 136
// CHECK-LP64-NEXT .quad   (__ZTC8test19_D0_9test19_B3) + 104
// CHECK-LP64-NEXT .quad   (__ZTC8test19_D0_9test19_B3) + 104
// CHECK-LP64-NEXT .quad   (__ZTC8test19_D0_9test19_B4) + 136
// CHECK-LP64-NEXT .quad   (__ZTC8test19_D0_9test19_B2) + 88
// CHECK-LP64-NEXT .quad   (__ZTC8test19_D0_9test19_B1) + 24


class test20_V {
  virtual void foo1();
};
class test20_V1 {
  virtual void foo2();
};
class test20_B : virtual test20_V {
} b;
class test20_B1 : virtual test20_V1 {
};
class test20_D : public test20_B, public test20_B1 {
};

// CHECK-LP64: __ZTV8test20_D:
// CHECK-LP64-NEXT: .quad 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI8test20_D
// CHECK-LP64-NEXT: .quad __ZN8test20_V4foo1Ev
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551608
// CHECK-LP64-NEXT: .quad __ZTI8test20_D
// CHECK-LP64-NEXT: .quad __ZN9test20_V14foo2Ev

// CHECK-LP64:     __ZTC8test20_D0_8test20_B:
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .quad   __ZTI8test20_B
// CHECK-LP64-NEXT: .quad   __ZN8test20_V4foo1Ev

// CHECK-LP64:     __ZTC8test20_D8_9test20_B1:
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .quad   __ZTI9test20_B1
// CHECK-LP64-NEXT: .quad   __ZN9test20_V14foo2Ev

// CHECK-LP64:     __ZTT8test20_D:
// CHECK-LP64-NEXT: .quad   (__ZTV8test20_D) + 40
// CHECK-LP64-NEXT: .quad   (__ZTC8test20_D0_8test20_B) + 32
// CHECK-LP64-NEXT: .quad   (__ZTC8test20_D0_8test20_B) + 32
// CHECK-LP64-NEXT: .quad   (__ZTC8test20_D8_9test20_B1) + 32
// CHECK-LP64-NEXT: .quad   (__ZTC8test20_D8_9test20_B1) + 32
// CHECK-LP64-NEXT .quad   (__ZTV8test20_D) + 40
// CHECK-LP64-NEXT .quad   (__ZTV8test20_D) + 80
// CHECK-LP64-NEXT .quad   (__ZTV8test20_D) + 80


class test21_V {
  virtual void foo() { }
};
class test21_V1 {
  virtual void foo() { }
};
class test21_B : virtual test21_V {
};
class test21_B1 : virtual test21_V1 {
};
class test21_D : public test21_B, public test21_B1 {
  void foo() { }
};

// CHECK-LP64: __ZTV8test21_D:
// CHECK-LP64-NEXT: .quad 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad __ZTI8test21_D
// CHECK-LP64-NEXT: .quad __ZN8test21_D3fooEv
// CHECK-LP64-NEXT: .space 8
// CHECK-LP64-NEXT: .quad 18446744073709551608
// CHECK-LP64-NEXT: .quad 18446744073709551608
// CHECK-LP64-NEXT: .quad __ZTI8test21_D
// CHECK-LP64-NEXT: .quad __ZTv0_n24_N8test21_D3fooEv

// CHECK-LP64:     __ZTC8test21_D0_8test21_B:
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .quad   __ZTI8test21_B
// CHECK-LP64-NEXT: .quad   __ZN8test21_V3fooEv

// CHECK-LP64:     __ZTC8test21_D8_9test21_B1:
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .space  8
// CHECK-LP64-NEXT: .quad   __ZTI9test21_B1
// CHECK-LP64-NEXT: .quad   __ZN9test21_V13fooEv

// CHECK-LP64:     __ZTT8test21_D:
// CHECK-LP64-NEXT: .quad   (__ZTV8test21_D) + 40
// CHECK-LP64-NEXT: .quad   (__ZTC8test21_D0_8test21_B) + 32
// CHECK-LP64-NEXT: .quad   (__ZTC8test21_D0_8test21_B) + 32
// CHECK-LP64-NEXT: .quad   (__ZTC8test21_D8_9test21_B1) + 32
// CHECK-LP64-NEXT: .quad   (__ZTC8test21_D8_9test21_B1) + 32
// CHECK-LP64-NEXT .quad   (__ZTV8test21_D) + 40
// CHECK-LP64-NEXT .quad   (__ZTV8test21_D) + 80
// CHECK-LP64-NEXT .quad   (__ZTV8test21_D) + 80



test21_D d21;
test20_D d20;
test19_D d19;
test18_D d18;
test17_D d17;
test16_D d16;
test15_D d15;
test13_D d13;
test11_D d11;
test10_D d10;
test9_D d9;
test8_D d8;

test5_D d5;
test4_D d4;
test3_D d3;

test6_D d6;
test7_D d7;
