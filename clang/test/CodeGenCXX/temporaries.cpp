// RUN: clang-cc -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s
struct A {
  A();
  ~A();
  void f();
};

void f1() {
  // CHECK: call void @_ZN1AC1Ev
  // CHECK: call void @_ZN1AD1Ev
  (void)A();

  // CHECK: call void @_ZN1AC1Ev
  // CHECK: call void @_ZN1AD1Ev
  A().f();
}

// Function calls
struct B {
  B();
  ~B();
};

B g();

void f2() {
  // CHECK-NOT: call void @_ZN1BC1Ev
  // CHECK: call void @_ZN1BD1Ev
  (void)g();
}

// Member function calls
struct C {
  C();
  ~C();
  
  C f();
};

void f3() {
  // CHECK: call void @_ZN1CC1Ev
  // CHECK: call void @_ZN1CD1Ev
  // CHECK: call void @_ZN1CD1Ev
  C().f();
}

// Function call operator
struct D {
  D();
  ~D();
  
  D operator()();
};

void f4() {
  // CHECK: call void @_ZN1DC1Ev
  // CHECK: call void @_ZN1DD1Ev
  // CHECK: call void @_ZN1DD1Ev
  D()();
}

// Overloaded operators
struct E {
  E();
  ~E();
  E operator+(const E&);
  E operator!();
};

void f5() {
  // CHECK: call void @_ZN1EC1Ev
  // CHECK: call void @_ZN1EC1Ev
  // CHECK: call void @_ZN1ED1Ev
  // CHECK: call void @_ZN1ED1Ev
  // CHECK: call void @_ZN1ED1Ev
  E() + E();
  
  // CHECK: call void @_ZN1EC1Ev
  // CHECK: call void @_ZN1ED1Ev
  // CHECK: call void @_ZN1ED1Ev
  !E();
}

struct F {
  F();
  ~F();
  F& f();
};

void f6() {
  // CHECK: call void @_ZN1FC1Ev
  // CHECK: call void @_ZN1FD1Ev
  F().f();
}

struct G {
  G();
  G(A);
  ~G();
  operator A();
};

void a(const A&);

void f7() {
  // CHECK: call void @_ZN1AC1Ev
  // CHECK: call void @_Z1aRK1A
  // CHECK: call void @_ZN1AD1Ev
  a(A());
  
  // CHECK: call void @_ZN1GC1Ev
  // CHECK: call void @_ZN1Gcv1AEv
  // CHECK: call void @_Z1aRK1A
  // CHECK: call void @_ZN1AD1Ev
  // CHECK: call void @_ZN1GD1Ev
  a(G());
}

namespace PR5077 {

struct A {
  A();
  ~A();
  int f();
};

void f();
int g(const A&);

struct B {
  int a1;
  int a2;
  B();
  ~B();
};

B::B()
  // CHECK: call void @_ZN6PR50771AC1Ev
  // CHECK: call i32 @_ZN6PR50771A1fEv
  // CHECK: call void @_ZN6PR50771AD1Ev
  : a1(A().f())
  // CHECK: call void @_ZN6PR50771AC1Ev
  // CHECK: call i32 @_ZN6PR50771gERKNS_1AE
  // CHECK: call void @_ZN6PR50771AD1Ev
  , a2(g(A()))
{
  // CHECK: call void @_ZN6PR50771fEv
  f();
}
  
struct C {
  C();
  
  const B& b;
};

C::C() 
  // CHECK: call void @_ZN6PR50771BC1Ev
  : b(B()) {
  // CHECK: call void @_ZN6PR50771fEv
  f();
  
  // CHECK: call void @_ZN6PR50771BD1Ev
}
}

A f8() {
  // CHECK: call void @_ZN1AC1Ev
  // CHECK-NOT: call void @_ZN1AD1Ev
  return A();
  // CHECK: ret void
}

struct H {
  H();
  ~H();
  H(const H&);
};

void f9(H h) {
  // CHECK: call void @_ZN1HC1Ev
  // CHECK: call void @_Z2f91H
  // CHECK: call void @_ZN1HD1Ev
  f9(H());
  
  // CHECK: call void @_ZN1HC1ERKS_
  // CHECK: call void @_Z2f91H
  // CHECK: call void @_ZN1HD1Ev
  f9(h);
}

void f10(const H&);

void f11(H h) {
  // CHECK: call void @_ZN1HC1Ev
  // CHECK: call void @_Z3f10RK1H
  // CHECK: call void @_ZN1HD1Ev
  f10(H());
  
  // CHECK: call void @_Z3f10RK1H
  // CHECK-NOT: call void @_ZN1HD1Ev
  // CHECK: ret void
  f10(h);
}