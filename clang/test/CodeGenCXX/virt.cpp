// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++0x -O0 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s

// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++0x -emit-llvm %s -o %t-64.ll
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

// CHECK-LPLL64:@_ZTV1B = constant [4 x i8*] [i8* null, i8* bitcast (%0* @_ZTI1B to i8*), i8* bitcast (void (%struct.B*)* @_ZN1B4bar1Ev to i8*), i8* bitcast (void (%struct.B*)* @_ZN1B4bar2Ev to i8*)]

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

// CHECK-LPLL64:@_ZTV1F = constant [19 x i8*] [i8* null, i8* inttoptr (i64 16 to i8*), i8* null, i8* null, i8* bitcast (%1* @_ZTI1F to i8*), i8* bitcast (void (%class.test14*)* @_ZN1D3booEv to i8*), i8* bitcast (void (%class.F*)* @_ZN1F3fooEv to i8*), i8* null, i8* null, i8* null, i8* null, i8* null, i8* inttoptr (i64 -16 to i8*), i8* bitcast (%1* @_ZTI1F to i8*), i8* bitcast (void (%struct.D1*)* @_ZN2D13barEv to i8*), i8* bitcast (void (%struct.D1*)* @_ZN2D14bar2Ev to i8*), i8* bitcast (void (%struct.D1*)* @_ZN2D14bar3Ev to i8*), i8* bitcast (void (%struct.D1*)* @_ZN2D14bar4Ev to i8*), i8* bitcast (void (%struct.D1*)* @_ZN2D14bar5Ev to i8*)]


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

// CHECK-LPLL64:@_ZTV1A = constant [10 x i8*] [i8* null, i8* bitcast (%2* @_ZTI1A to i8*), i8* bitcast (void (%struct.B*)* @_ZN1B4bar1Ev to i8*), i8* bitcast (void (%struct.B*)* @_ZN1B4bar2Ev to i8*), i8* bitcast (void (%class.A*)* @_ZN1A4foo1Ev to i8*), i8* bitcast (void (%class.A*)* @_ZN1A4foo2Ev to i8*), i8* inttoptr (i64 -16 to i8*), i8* bitcast (%2* @_ZTI1A to i8*), i8* bitcast (void (%class.test14*)* @_ZN1C4bee1Ev to i8*), i8* bitcast (void (%class.test14*)* @_ZN1C4bee2Ev to i8*)]

int main() {
  A a;
  B b;
  ap->e = 1;
  ap->b = 2;
}


struct test12_A {
  virtual void foo0() { }
  virtual void foo();
} *test12_pa;

struct test12_B : public test12_A {
  virtual void foo() { }
} *test12_pb;

struct test12_D : public test12_B {
} *test12_pd;


struct test6_B2 { virtual void funcB2(); char b[1000]; };
struct test6_B1 : virtual test6_B2 { virtual void funcB1(); };

struct test6_D : test6_B2, virtual test6_B1 {
};

// CHECK-LP64: .zerofill __DATA,__common,_d6,2024,4

struct test7_B2 { virtual void funcB2(); };
struct test7_B1 : virtual test7_B2 { virtual void funcB1(); };

struct test7_D : test7_B2, virtual test7_B1 {
};

// CHECK-LP64: .zerofill __DATA,__common,_d7,16,3


struct test3_B3 { virtual void funcB3(); };
struct test3_B2 : virtual test3_B3 { virtual void funcB2(); };
struct test3_B1 : virtual test3_B2 { virtual void funcB1(); };

struct test3_D : virtual test3_B1 {
  virtual void funcD() { }
};

// CHECK-LPLL64:@_ZTV7test3_D = weak_odr constant [12 x i8*] [i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* bitcast (%3* @_ZTI7test3_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN8test3_B36funcB3Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN8test3_B26funcB2Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN8test3_B16funcB1Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN7test3_D5funcDEv to i8*)]


struct test4_D : virtual B, virtual C {
};

// CHECK-LPLL64:@_ZTV7test4_D = weak_odr constant [14 x i8*] [i8* null, i8* inttoptr (i64 8 to i8*), i8* null, i8* null, i8* null, i8* bitcast (%1* @_ZTI7test4_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN1C4bee1Ev to i8*), i8* bitcast (void (%class.test14*)* @_ZN1C4bee2Ev to i8*), i8* null, i8* null, i8* inttoptr (i64 -8 to i8*), i8* bitcast (%1* @_ZTI7test4_D to i8*), i8* bitcast (void (%struct.B*)* @_ZN1B4bar1Ev to i8*), i8* bitcast (void (%struct.B*)* @_ZN1B4bar2Ev to i8*)]


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

// CHECK-LPLL64:@_ZTV7test5_D = weak_odr constant [50 x i8*] [i8* inttoptr (i64 32 to i8*), i8* inttoptr (i64 24 to i8*), i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 8 to i8*), i8* inttoptr (i64 8 to i8*), i8* inttoptr (i64 8 to i8*), i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* bitcast (%2* @_ZTI7test5_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN8test5_B36funcB3Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN8test5_B26funcB2Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN8test5_B16funcB1Ev to i8*), i8* bitcast (void (%struct.test10_B2*)* @_ZN7test5_D5funcDEv to i8*), i8* null, i8* null, i8* null, i8* null, i8* null, i8* inttoptr (i64 -8 to i8*), i8* bitcast (%2* @_ZTI7test5_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN9test5_B237funcB23Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test5_B227funcB22Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test5_B217funcB21Ev to i8*), i8* null, i8* inttoptr (i64 16 to i8*), i8* null, i8* null, i8* inttoptr (i64 8 to i8*), i8* null, i8* null, i8* inttoptr (i64 -16 to i8*), i8* bitcast (%2* @_ZTI7test5_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN9test5_B337funcB33Ev to i8*), i8* bitcast (void (%class.test20_D*)* @_ZN9test5_B327funcB32Ev to i8*), i8* bitcast (void (%class.test23_D*)* @_ZN9test5_B317funcB31Ev to i8*), i8* null, i8* inttoptr (i64 -24 to i8*), i8* bitcast (%2* @_ZTI7test5_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN4B2328funcB232Ev to i8*), i8* null, i8* inttoptr (i64 -32 to i8*), i8* bitcast (%2* @_ZTI7test5_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN4B2318funcB231Ev to i8*)]

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

// CHECK-LPLL64:@_ZTV7test8_D = weak_odr constant [25 x i8*] [i8* inttoptr (i64 48 to i8*), i8* inttoptr (i64 32 to i8*), i8* null, i8* bitcast (%2* @_ZTI7test8_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN8test8_B19ftest8_B1Ev to i8*), i8* inttoptr (i64 40 to i8*), i8* inttoptr (i64 24 to i8*), i8* inttoptr (i64 -8 to i8*), i8* bitcast (%2* @_ZTI7test8_D to i8*), i8* bitcast (void (%struct.test10_B2a*)* @_ZN9test8_B2a10ftest8_B2aEv to i8*), i8* bitcast (void (%struct.test15_D*)* @_ZN8test8_B29ftest8_B2Ev to i8*), i8* inttoptr (i64 -16 to i8*), i8* bitcast (%2* @_ZTI7test8_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN9test8_B2b10ftest8_B2bEv to i8*), i8* inttoptr (i64 -24 to i8*), i8* bitcast (%2* @_ZTI7test8_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN8test8_B39ftest8_B3Ev to i8*), i8* null, i8* inttoptr (i64 -32 to i8*), i8* bitcast (%2* @_ZTI7test8_D to i8*), i8* bitcast (void (%struct.B*)* @_ZN10test8_B2aa11ftest8_B2aaEv to i8*), i8* null, i8* inttoptr (i64 -48 to i8*), i8* bitcast (%2* @_ZTI7test8_D to i8*), i8* bitcast (void (%struct.B*)* @_ZN10test8_B2ab11ftest8_B2abEv to i8*)]

// CHECK-LPLL64:@_ZTC7test8_D8_8test8_B2 = internal constant [14 x i8*] [i8* inttoptr (i64 40 to i8*), i8* inttoptr (i64 24 to i8*), i8* null, i8* bitcast (%1* @_ZTI8test8_B2 to i8*), i8* bitcast (void (%struct.test10_B2a*)* @_ZN9test8_B2a10ftest8_B2aEv to i8*), i8* bitcast (void (%struct.test15_D*)* @_ZN8test8_B29ftest8_B2Ev to i8*), i8* null, i8* inttoptr (i64 -24 to i8*), i8* bitcast (%1* @_ZTI8test8_B2 to i8*), i8* bitcast (void (%struct.B*)* @_ZN10test8_B2aa11ftest8_B2aaEv to i8*), i8* null, i8* inttoptr (i64 -40 to i8*), i8* bitcast (%1* @_ZTI8test8_B2 to i8*), i8* bitcast (void (%struct.B*)* @_ZN10test8_B2ab11ftest8_B2abEv to i8*)] ; <[14 x i8*]*> [#uses=3]

// CHECK-LPLL64:@_ZTC7test8_D8_9test8_B2a = internal constant [13 x i8*] [i8* inttoptr (i64 40 to i8*), i8* inttoptr (i64 24 to i8*), i8* null, i8* bitcast (%1* @_ZTI9test8_B2a to i8*), i8* bitcast (void (%struct.test10_B2a*)* @_ZN9test8_B2a10ftest8_B2aEv to i8*), i8* null, i8* inttoptr (i64 -24 to i8*), i8* bitcast (%1* @_ZTI9test8_B2a to i8*), i8* bitcast (void (%struct.B*)* @_ZN10test8_B2aa11ftest8_B2aaEv to i8*), i8* null, i8* inttoptr (i64 -40 to i8*), i8* bitcast (%1* @_ZTI9test8_B2a to i8*), i8* bitcast (void (%struct.B*)* @_ZN10test8_B2ab11ftest8_B2abEv to i8*)] ; <[13 x i8*]*> [#uses=3]

// CHECK-LPLL64:@_ZTT7test8_D = weak_odr constant [10 x i8*] [i8* bitcast (i8** getelementptr inbounds ([25 x i8*]* @_ZTV7test8_D, i64 0, i64 4) to i8*), i8* bitcast (i8** getelementptr inbounds ([14 x i8*]* @_ZTC7test8_D8_8test8_B2, i64 0, i64 4) to i8*), i8* bitcast (i8** getelementptr inbounds ([13 x i8*]* @_ZTC7test8_D8_9test8_B2a, i64 0, i64 4) to i8*), i8* bitcast (i8** getelementptr inbounds ([13 x i8*]* @_ZTC7test8_D8_9test8_B2a, i64 0, i64 8) to i8*), i8* bitcast (i8** getelementptr inbounds ([13 x i8*]* @_ZTC7test8_D8_9test8_B2a, i64 0, i64 12) to i8*), i8* bitcast (i8** getelementptr inbounds ([14 x i8*]* @_ZTC7test8_D8_8test8_B2, i64 0, i64 9) to i8*), i8* bitcast (i8** getelementptr inbounds ([14 x i8*]* @_ZTC7test8_D8_8test8_B2, i64 0, i64 13) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*]* @_ZTV7test8_D, i64 0, i64 9) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*]* @_ZTV7test8_D, i64 0, i64 20) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*]* @_ZTV7test8_D, i64 0, i64 24) to i8*)]


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

// CHECK-LPLL64:@_ZTV7test9_D = weak_odr constant [70 x i8*] [i8* inttoptr (i64 168 to i8*), i8* inttoptr (i64 152 to i8*), i8* inttoptr (i64 136 to i8*), i8* inttoptr (i64 120 to i8*), i8* inttoptr (i64 104 to i8*), i8* inttoptr (i64 88 to i8*), i8* inttoptr (i64 72 to i8*), i8* inttoptr (i64 56 to i8*), i8* inttoptr (i64 40 to i8*), i8* inttoptr (i64 24 to i8*), i8* inttoptr (i64 8 to i8*), i8* null, i8* bitcast (%2* @_ZTI7test9_D to i8*), i8* bitcast (void (%struct.test9_D*)* @_ZN7test9_D5funcDEv to i8*), i8* null, i8* inttoptr (i64 32 to i8*), i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 -8 to i8*), i8* bitcast (%2* @_ZTI7test9_D to i8*), i8* bitcast (void (%struct.test9_B1*)* @_ZN8test9_B16funcB1Ev to i8*), i8* null, i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 -24 to i8*), i8* bitcast (%2* @_ZTI7test9_D to i8*), i8* bitcast (void (%struct.test13_B2*)* @_ZN8test9_B26funcB2Ev to i8*), i8* null, i8* inttoptr (i64 -40 to i8*), i8* bitcast (%2* @_ZTI7test9_D to i8*), i8* bitcast (void (%struct.B*)* @_ZN8test9_B36funcB3Ev to i8*), i8* null, i8* inttoptr (i64 32 to i8*), i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 -56 to i8*), i8* bitcast (%2* @_ZTI7test9_D to i8*), i8* bitcast (void (%struct.test9_B1*)* @_ZN9test9_B217funcB21Ev to i8*), i8* null, i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 -72 to i8*), i8* bitcast (%2* @_ZTI7test9_D to i8*), i8* bitcast (void (%struct.test13_B2*)* @_ZN9test9_B227funcB22Ev to i8*), i8* null, i8* inttoptr (i64 -88 to i8*), i8* bitcast (%2* @_ZTI7test9_D to i8*), i8* bitcast (void (%struct.B*)* @_ZN9test9_B237funcB23Ev to i8*), i8* null, i8* inttoptr (i64 64 to i8*), i8* inttoptr (i64 48 to i8*), i8* inttoptr (i64 32 to i8*), i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 -104 to i8*), i8* bitcast (%2* @_ZTI7test9_D to i8*), i8* bitcast (void (%struct.test9_B31*)* @_ZN9test9_B317funcB31Ev to i8*), i8* null, i8* inttoptr (i64 32 to i8*), i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 -120 to i8*), i8* bitcast (%2* @_ZTI7test9_D to i8*), i8* bitcast (void (%struct.test9_B1*)* @_ZN9test9_B327funcB32Ev to i8*), i8* null, i8* inttoptr (i64 -136 to i8*), i8* bitcast (%2* @_ZTI7test9_D to i8*), i8* bitcast (void (%struct.B*)* @_ZN9test9_B337funcB33Ev to i8*), i8* null, i8* inttoptr (i64 -152 to i8*), i8* bitcast (%2* @_ZTI7test9_D to i8*), i8* bitcast (void (%struct.B*)* @_ZN10test9_B2328funcB232Ev to i8*), i8* null, i8* inttoptr (i64 -168 to i8*), i8* bitcast (%2* @_ZTI7test9_D to i8*), i8* bitcast (void (%struct.B*)* @_ZN10test9_B2318funcB231Ev to i8*)] ; <[70 x i8*]*> [#uses=12]


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

// CHECK-LPLL64:@_ZTV8test10_D = weak_odr constant [19 x i8*] [i8* inttoptr (i64 40 to i8*), i8* inttoptr (i64 24 to i8*), i8* inttoptr (i64 16 to i8*), i8* null, i8* bitcast (%1* @_ZTI8test10_D to i8*), i8* bitcast (void (%struct.test10_B1*)* @_ZN9test10_B110ftest10_B1Ev to i8*), i8* inttoptr (i64 32 to i8*), i8* inttoptr (i64 8 to i8*), i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 -8 to i8*), i8* bitcast (%1* @_ZTI8test10_D to i8*), i8* bitcast (void (%struct.test10_B2a*)* @_ZN10test10_B2a11ftest10_B2aEv to i8*), i8* bitcast (void (%struct.test10_B2*)* @_ZN9test10_B210ftest10_B2Ev to i8*), i8* inttoptr (i64 -8 to i8*), i8* inttoptr (i64 -24 to i8*), i8* bitcast (%1* @_ZTI8test10_D to i8*), i8* inttoptr (i64 -24 to i8*), i8* inttoptr (i64 -40 to i8*), i8* bitcast (%1* @_ZTI8test10_D to i8*)] ; <[19 x i8*]*> [#uses=4]


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

// CHECK-LPLL64:@_ZTV8test11_D = weak_odr constant [7 x i8*] [i8* null, i8* bitcast (%4* @_ZTI8test11_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN8test11_B2B1Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN8test11_D1DEv to i8*), i8* bitcast (void (%class.test14*)* @_ZN8test11_B2B2Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN8test11_D2D1Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN8test11_D2D2Ev to i8*)]


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

// CHECK-LPLL64:@_ZTV8test13_D = weak_odr constant [39 x i8*] [i8* inttoptr (i64 24 to i8*), i8* inttoptr (i64 8 to i8*), i8* null, i8* bitcast (%1* @_ZTI8test13_D to i8*), i8* bitcast (void (%struct.test10_B2*)* @_ZN8test13_D6fooNV1Ev to i8*), i8* bitcast (void (%struct.test10_B2*)* @_ZN8test13_D1DEv to i8*), i8* bitcast (void (%struct.test10_B2*)* @_ZN8test13_D2D1Ev to i8*), i8* bitcast (void (%struct.test10_B2*)* @_ZN8test13_D2DbEv to i8*), i8* bitcast (void (%struct.test10_B2*)* @_ZN8test13_D2DdEv to i8*), i8* bitcast (void (%struct.test10_B2*)* @_ZN8test13_D2D2Ev to i8*), i8* null, i8* inttoptr (i64 -8 to i8*), i8* null, i8* inttoptr (i64 -8 to i8*), i8* null, i8* null, i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 -8 to i8*), i8* bitcast (%1* @_ZTI8test13_D to i8*), i8* bitcast (void (%struct.test13_B2*)* @_ZN9test13_B23B2aEv to i8*), i8* bitcast (void (%struct.test13_B2*)* @_ZN9test13_B22B2Ev to i8*), i8* bitcast (void (%struct.test10_B2*)* @_ZTv0_n48_N8test13_D1DEv to i8*), i8* bitcast (void (%struct.test13_B2*)* @_ZN9test13_B22DaEv to i8*), i8* bitcast (void (%struct.test10_B2*)* @_ZTv0_n64_N8test13_D2DdEv to i8*), i8* bitcast (void (%struct.test13_B2*)* @_ZN9test13_B23B2bEv to i8*), i8* inttoptr (i64 -16 to i8*), i8* null, i8* inttoptr (i64 -24 to i8*), i8* inttoptr (i64 -16 to i8*), i8* inttoptr (i64 -24 to i8*), i8* null, i8* inttoptr (i64 -24 to i8*), i8* bitcast (%1* @_ZTI8test13_D to i8*), i8* bitcast (void (%struct.B*)* @_ZN8test13_B2B1Ev to i8*), i8* bitcast (void (%struct.test10_B2*)* @_ZTv0_n32_N8test13_D1DEv to i8*), i8* bitcast (void (%struct.test13_B2*)* @_ZTv0_n40_N9test13_B22DaEv to i8*), i8* bitcast (void (%struct.test10_B2*)* @_ZTv0_n48_N8test13_D2DbEv to i8*), i8* bitcast (void (%struct.B*)* @_ZN8test13_B2DcEv to i8*), i8* bitcast (void (%struct.test13_B2*)* @_ZTv0_n64_N9test13_B22B2Ev to i8*)]


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

// CHECK-LPLL64:@_ZTV8test15_D = weak_odr constant [23 x i8*] [i8* inttoptr (i64 32 to i8*), i8* inttoptr (i64 16 to i8*), i8* null, i8* bitcast (%1* @_ZTI8test15_D to i8*), i8* bitcast (void (%struct.B*)* @_ZN10test15_NV16fooNV1Ev to i8*), i8* bitcast (%struct.test15_D* (%struct.test15_D*)* @_ZN8test15_D4foo1Ev to i8*), i8* null, i8* inttoptr (i64 -16 to i8*), i8* null, i8* inttoptr (i64 16 to i8*), i8* inttoptr (i64 -16 to i8*), i8* bitcast (%1* @_ZTI8test15_D to i8*), i8* bitcast (void (%struct.B*)* @_ZN10test15_NV16fooNV1Ev to i8*), i8* bitcast (%struct.test15_D* (%struct.test15_D*)* @_ZTcv0_n40_v0_n24_N8test15_D4foo1Ev to i8*), i8* bitcast (%struct.test15_B2* (%struct.test15_B2*)* @_ZN9test15_B24foo2Ev to i8*), i8* null, i8* inttoptr (i64 -16 to i8*), i8* inttoptr (i64 -32 to i8*), i8* inttoptr (i64 -32 to i8*), i8* bitcast (%1* @_ZTI8test15_D to i8*), i8* bitcast (%struct.test15_D* (%struct.test15_D*)* @_ZTcv0_n24_v0_n32_N8test15_D4foo1Ev to i8*), i8* bitcast (%struct.test15_B2* (%struct.test15_B2*)* @_ZTcv0_n32_v0_n24_N9test15_B24foo2Ev to i8*), i8* bitcast (%struct.B* (%struct.B*)* @_ZN8test15_B4foo3Ev to i8*)]


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

// FIXME:
// CHECK-LP64: __ZTV8test16_D:
// CHECK-LP64-NEXT: .quad 32
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 0
// CHECK-LP64-NEXT: .quad __ZTI8test16_D
// CHECK-LP64-NEXT: .quad __ZN10test16_NV16fooNV1Ev
// CHECK-LP64-NEXT: .quad __ZN10test16_NV17foo_NV1Ev
// CHECK-LP64-NEXT: .quad __ZN8test16_D3barEv
// CHECK-LP64-NEXT: .quad __ZN8test16_D4foo1Ev
// CHECK-LP64-NEXT: .quad 0
// CHECK-LP64-NEXT: .quad 0
// CHECK-LP64-NEXT: .quad -16
// CHECK-LP64-NEXT: .quad 0
// CHECK-LP64-NEXT: .quad 0
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad -16
// CHECK-LP64-NEXT: .quad __ZTI8test16_D
// CHECK-LP64-NEXT: .quad __ZN10test16_NV16fooNV1Ev
// CHECK-LP64-NEXT: .quad __ZN10test16_NV17foo_NV1Ev
// CHECK-LP64-NEXT: .quad __ZTcv0_n48_v0_n24_N8test16_D4foo1Ev
// CHECK-LP64-NEXT: .quad __ZN9test16_B24foo2Ev
// CHECK-LP64-NEXT: .quad __ZN9test16_B26foo_B2Ev
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 16
// CHECK-LP64-NEXT: .quad 0
// CHECK-LP64-NEXT: .quad 0
// CHECK-LP64-NEXT: .quad -16
// CHECK-LP64-NEXT: .quad -32
// CHECK-LP64-NEXT: .quad 0
// CHECK-LP64-NEXT: .quad 0
// CHECK-LP64-NEXT: .quad -32
// CHECK-LP64-NEXT: .quad __ZTI8test16_D
// CHECK-LP64-NEXT: .quad __ZN10test16_NV16fooNV1Ev
// CHECK-LP64-NEXT: .quad __ZN10test16_NV17foo_NV1Ev
// CHECK-LP64-NEXT: .quad __ZTcv0_n40_v0_n32_N8test16_D4foo1Ev
// CHECK-LP64-NEXT: .quad __ZTcv0_n48_v0_n24_N9test16_B24foo2Ev
// CHECK-LP64-NEXT: .quad __ZN8test16_B4foo3Ev
// CHECK-LP64-NEXT: .quad __ZN8test16_B5foo_BEv
// CHECK-LP64-NEXT: .quad -48
// CHECK-LP64-NEXT: .quad __ZTI8test16_D
// CHECK-LP64-NEXT: .quad __ZTcvn16_n40_v16_n32_N8test16_D4foo1Ev
// CHECK-LP64-NEXT: .quad __ZN10test16_NV27foo_NV2Ev
// CHECK-LP64-NEXT: .quad __ZN10test16_NV28foo_NV2bEv




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


// CHECK-LPLL64:@_ZTV8test17_D = weak_odr constant [4 x i8*] [i8* null, i8* bitcast (%4* @_ZTI8test17_D to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test17_B23fooEv to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN8test17_D3barEv to i8*)]

// CHECK-LPLL64:@_ZTV9test17_B2 = weak_odr constant [4 x i8*] [i8* null, i8* bitcast (%4* @_ZTI9test17_B2 to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test17_B23fooEv to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*)]

// CHECK-LPLL64:@_ZTV9test17_B1 = weak_odr constant [4 x i8*] [i8* null, i8* bitcast (%0* @_ZTI9test17_B1 to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*), i8* bitcast (void (%class.test14*)* @_ZN9test17_B13barEv to i8*)]


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


// CHECK-LPLL64:@_ZTV8test19_D = weak_odr constant [28 x i8*] [i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* bitcast (%3* @_ZTI8test19_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN9test19_B13fB1Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test19_B26foB1B2Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test19_B36foB1B3Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test19_B46foB1B4Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test19_B23fB2Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test19_B36foB2B3Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test19_B46foB2B4Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test19_B33fB3Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test19_B46foB3B4Ev to i8*), i8* bitcast (void (%class.test17_B2*)* @_ZN9test19_B43fB4Ev to i8*)]

// FIXME:
// CHECK-LP64:     __ZTT8test19_D:
// CHECK-LP64-NEXT: .quad __ZTV8test19_D+144
// CHECK-LP64-NEXT: .quad __ZTV8test19_D+144
// CHECK-LP64-NEXT .quad __ZTV8test19_D+144
// CHECK-LP64-NEXT .quad __ZTC8test19_D0_9test19_B4+136
// CHECK-LP64-NEXT .quad __ZTC8test19_D0_9test19_B3+104
// CHECK-LP64-NEXT .quad __ZTC8test19_D0_9test19_B3+104
// CHECK-LP64-NEXT .quad __ZTC8test19_D0_9test19_B4+136
// CHECK-LP64-NEXT .quad __ZTC8test19_D0_9test19_B2+88
// CHECK-LP64-NEXT .quad __ZTC8test19_D0_9test19_B1+24

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

// CHECK-LPLL64:@_ZTV8test20_D = weak_odr constant [11 x i8*] [i8* inttoptr (i64 8 to i8*), i8* null, i8* null, i8* null, i8* bitcast (%1* @_ZTI8test20_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN8test20_V4foo1Ev to i8*), i8* null, i8* null, i8* inttoptr (i64 -8 to i8*), i8* bitcast (%1* @_ZTI8test20_D to i8*), i8* bitcast (void (%class.test14*)* @_ZN9test20_V14foo2Ev to i8*)]

// CHECK-LPLL64:@_ZTC8test20_D0_8test20_B = internal constant [5 x i8*] [i8* null, i8* null, i8* null, i8* bitcast (%3* @_ZTI8test20_B to i8*), i8* bitcast (void (%class.test14*)* @_ZN8test20_V4foo1Ev to i8*)]

// CHECK-LPLL64:@_ZTC8test20_D8_9test20_B1 = internal constant [5 x i8*] [i8* null, i8* null, i8* null, i8* bitcast (%3* @_ZTI9test20_B1 to i8*), i8* bitcast (void (%class.test14*)* @_ZN9test20_V14foo2Ev to i8*)] ; <[5 x i8*]*> [#uses=1]

// FIXME: 
// CHECK-LP64:     __ZTT8test20_D:
// CHECK-LP64-NEXT: .quad __ZTV8test20_D+40
// CHECK-LP64-NEXT: .quad __ZTC8test20_D0_8test20_B+32
// CHECK-LP64-NEXT: .quad __ZTC8test20_D0_8test20_B+32
// CHECK-LP64-NEXT: .quad __ZTC8test20_D8_9test20_B1+32
// CHECK-LP64-NEXT: .quad __ZTC8test20_D8_9test20_B1+32
// CHECK-LP64-NEXT .quad __ZTV8test20_D+40
// CHECK-LP64-NEXT .quad __ZTV8test20_D+80
// CHECK-LP64-NEXT .quad __ZTV8test20_D+80


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

// CHECK-LPLL64:@_ZTV8test21_D = weak_odr constant [11 x i8*] [i8* inttoptr (i64 8 to i8*), i8* null, i8* null, i8* null, i8* bitcast (%1* @_ZTI8test21_D to i8*), i8* bitcast (void (%class.test20_D*)* @_ZN8test21_D3fooEv to i8*), i8* null, i8* inttoptr (i64 -8 to i8*), i8* inttoptr (i64 -8 to i8*), i8* bitcast (%1* @_ZTI8test21_D to i8*), i8* bitcast (void (%class.test20_D*)* @_ZTv0_n24_N8test21_D3fooEv to i8*)]

// CHECK-LPLL64:@_ZTC8test21_D0_8test21_B = internal constant [5 x i8*] [i8* null, i8* null, i8* null, i8* bitcast (%3* @_ZTI8test21_B to i8*), i8* bitcast (void (%class.test14*)* @_ZN8test21_V3fooEv to i8*)]

// CHECK-LPLL64:@_ZTC8test21_D8_9test21_B1 = internal constant [5 x i8*] [i8* null, i8* null, i8* null, i8* bitcast (%3* @_ZTI9test21_B1 to i8*), i8* bitcast (void (%class.test14*)* @_ZN9test21_V13fooEv to i8*)] ; <[5 x i8*]*> [#uses=1]

// FIXME:
// CHECK-LP64:     __ZTT8test21_D:
// CHECK-LP64-NEXT: .quad __ZTV8test21_D+40
// CHECK-LP64-NEXT: .quad __ZTC8test21_D0_8test21_B+32
// CHECK-LP64-NEXT: .quad __ZTC8test21_D0_8test21_B+32
// CHECK-LP64-NEXT: .quad __ZTC8test21_D8_9test21_B1+32
// CHECK-LP64-NEXT: .quad __ZTC8test21_D8_9test21_B1+32
// CHECK-LP64-NEXT .quad __ZTV8test21_D+40
// CHECK-LP64-NEXT .quad __ZTV8test21_D+80
// CHECK-LP64-NEXT .quad __ZTV8test21_D+80


struct test22_s1 { virtual void dtor() { } }; 
struct test22_s2 { virtual void dtor() { } }; 
struct test22_s3 : test22_s1, test22_s2 { virtual void dtor() { } }; 
struct test22_D : test22_s3 { virtual void dtor() { } }; 

// CHECK-LPLL64:@_ZTV8test22_D = weak_odr constant [6 x i8*] [i8* null, i8* bitcast (%4* @_ZTI8test22_D to i8*), i8* bitcast (void (%class.test20_D*)* @_ZN8test22_D4dtorEv to i8*), i8* inttoptr (i64 -8 to i8*), i8* bitcast (%4* @_ZTI8test22_D to i8*), i8* bitcast (void (%class.test20_D*)* @_ZThn8_N8test22_D4dtorEv to i8*)]


class test23_s1 {
  virtual void fun1(char *t) { }
};
class test23_s2 {
  virtual void fun2(char *t) { }
};
class test23_s3 {
  virtual void fun3(char *t) { }
};
class test23_s4: virtual test23_s1, test23_s2, test23_s3 {
  virtual void fun4(char *t) { }
};
class test23_D: virtual test23_s4 {
  virtual void fun5(char *t) { }
};


// FIXME:
// CHECK-LP64:     __ZTV8test23_D:
// CHECK-LP64-NEXT:	.quad	0
// CHECK-LP64-NEXT:	.quad	8
// CHECK-LP64-NEXT:	.quad	0
// CHECK-LP64-NEXT:	.quad	0
// CHECK-LP64-NEXT:	.quad	__ZTI8test23_D
// CHECK-LP64-NEXT:	.quad	__ZN9test23_s14fun1EPc
// CHECK-LP64-NEXT:	.quad	__ZN8test23_D4fun5EPc
// CHECK-LP64-NEXT	.quad	8
// CHECK-LP64:  	.quad	0
// CHECK-LP64-NEXT:	.quad	0
// CHECK-LP64:  	.quad	-8
// CHECK-LP64-NEXT:	.quad	-8
// CHECK-LP64-NEXT:	.quad	__ZTI8test23_D
// CHECK-LP64-NEXT:	.quad	__ZN9test23_s24fun2EPc
// CHECK-LP64-NEXT:	.quad	__ZN9test23_s44fun4EPc
// CHECK-LP64-NEXT:	.quad	-16
// CHECK-LP64-NEXT:	.quad	__ZTI8test23_D
// CHECK-LP64-NEXT:	.quad	__ZN9test23_s34fun3EPc


test23_D d23;
test22_D d22;
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
// CHECK-LPLL64:  call void %
// CHECK-LPLL64:  call void %
// CHECK-LPLL64:  call void %
// CHECK-LPLL64:  call void %
// CHECK-LPLL64:  call void %
// CHECK-LPLL64:  call void %
// CHECK-LPLL64:  call void @_ZN8test12_A3fooEv(%class.test14* %{{.*}})


// FIXME: This is the wrong thunk, but until these issues are fixed, better
// than nothing.
// CHECK-LPLL64define weak %class.test8_D* @_ZTcvn16_n72_v16_n32_N8test16_D4foo1Ev(%class.test8_D*)
// CHECK-LPLL64  %{{retval|2}} = alloca %class.test8_D*
// CHECK-LPLL64  %.addr = alloca %class.test8_D*
// CHECK-LPLL64  store %class.test8_D* %0, %class.test8_D** %.addr
// CHECK-LPLL64  %{{this|3}} = load %class.test8_D** %.addr
// CHECK-LPLL64  %{{1|4}} = bitcast %class.test8_D* %{{this|3}} to i8*
// CHECK-LPLL64  %{{2|5}} = getelementptr inbounds i8* %{{1|4}}, i64 -16
// CHECK-LPLL64  %{{3|6}} = bitcast i8* %{{2|5}} to %class.test8_D*
// CHECK-LPLL64  %{{4|7}} = bitcast %class.test8_D* %{{3|6}} to i8*
// CHECK-LPLL64  %{{5|8}} = bitcast %class.test8_D* %3 to i64**
// CHECK-LPLL64  %{{vtable|9}} = load i64** %{{5|8}}
// CHECK-LPLL64  %{{6|10}} = getelementptr inbounds i64* %{{vtable|9}}, i64 -9
// CHECK-LPLL64  %{{7|11}} = load i64* %{{6|10}}
// CHECK-LPLL64  %{{8|12}} = getelementptr i8* %{{4|7}}, i64 %{{7|11}}
// CHECK-LPLL64  %{{9|13}} = bitcast i8* %{{8|12}} to %class.test8_D*
// CHECK-LPLL64  %{{call|14}} = call %class.test8_D* @_ZTch0_v16_n32_N8test16_D4foo1Ev(%class.test8_D* %{{9|13}})
// CHECK-LPLL64  store %class.test8_D* %{{call|14}}, %class.test8_D** %{{retval|2}}
// CHECK-LPLL64  %{{10|15}} = load %class.test8_D** %{{retval|2}}
// CHECK-LPLL64  ret %class.test8_D* %{{10|15}}
// CHECK-LPLL64}

// CHECK-LPLL64:define weak %class.test8_D* @_ZTch0_v16_n32_N8test16_D4foo1Ev(%{{class.test8_D|.*}}*)
// CHECK-LPLL64:  %{{retval|2}} = alloca %class.test8_D*
// CHECK-LPLL64:  %.addr = alloca %class.test8_D*
// CHECK-LPLL64:  store %class.test8_D* %0, %class.test8_D** %.addr
// CHECK-LPLL64:  %{{this|3}} = load %class.test8_D** %.addr
// CHECK-LPLL64:  %{{call|4}} = call %class.test8_D* @_ZN8test16_D4foo1Ev(%class.test8_D* %{{this|3}})
// CHECK-LPLL64:  %{{1|5}} = icmp ne %class.test8_D* %{{call|4}}, null
// CHECK-LPLL64:  br i1 %{{1|5}}, label %{{2|6}}, label %{{12|17}}
// CHECK-LPLL64:; <label>:{{2|6}}
// CHECK-LPLL64:  %{{3|7}} = bitcast %class.test8_D* %{{call|4}} to i8*
// CHECK-LPLL64:  %{{4|8}} = getelementptr inbounds i8* %{{3|7}}, i64 16
// CHECK-LPLL64:  %{{5|9}} = bitcast i8* %4 to %class.test8_D*
// CHECK-LPLL64:  %{{6|10}} = bitcast %class.test8_D* %{{5|9}} to i8*
// CHECK-LPLL64:  %{{7|11}} = bitcast %class.test8_D* %{{5|9}} to i64**
// CHECK-LPLL64:  %{{vtable|12}} = load i64** %{{7|11}}
// CHECK-LPLL64:  %{{8|13}} = getelementptr inbounds i64* %vtable, i64 -4
// CHECK-LPLL64:  %{{9|14}} = load i64* %{{8|13}}
// CHECK-LPLL64:  %{{10|15}} = getelementptr i8* %{{6|10}}, i64 %{{9|14}}
// CHECK-LPLL64:  %{{11|16}} = bitcast i8* %{{10|15}} to %class.test8_D*
// CHECK-LPLL64:  br label %{{13|18}}
// CHECK-LPLL64:; <label>:{{12|17}}
// CHECK-LPLL64:  br label %{{13|18}}
// CHECK-LPLL64:; <label>:{{13|18}}
// CHECK-LPLL64:  %{{14|19}} = phi %class.test8_D* [ %{{11|16}}, %{{2|6}} ], [ %{{call|4}}, %{{12|17}} ]
// CHECK-LPLL64:  store %class.test8_D* %{{14|19}}, %class.test8_D** %{{retval|2}}
// CHECK-LPLL64:  %{{15|20}} = load %class.test8_D** %{{retval|2}}
// CHECK-LPLL64:  ret %class.test8_D* %{{15|20}}
// CHECK-LPLL64:}
