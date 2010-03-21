// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s
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

// PR5808
struct I {
  I(const char *);
  ~I();
};

// CHECK: _Z3f12v
I f12() {
  // CHECK: call void @_ZN1IC1EPKc
  // CHECK-NOT: call void @_ZN1ID1Ev
  // CHECK: ret void
  return "Hello";
}

// PR5867
namespace PR5867 {
  struct S {
    S();
    S(const S &);
    ~S();
  };

  void f(S, int);
  // CHECK: define void @_ZN6PR58671gEv
  void g() {
    // CHECK: call void @_ZN6PR58671SC1Ev
    // CHECK-NEXT: call void @_ZN6PR58671fENS_1SEi
    // CHECK-NEXT: call void @_ZN6PR58671SD1Ev
    // CHECK-NEXT: ret void
    (f)(S(), 0);
  }

  // CHECK: define linkonce_odr void @_ZN6PR58672g2IiEEvT_
  template<typename T>
  void g2(T) {
    // CHECK: call void @_ZN6PR58671SC1Ev
    // CHECK-NEXT: call void @_ZN6PR58671fENS_1SEi
    // CHECK-NEXT: call void @_ZN6PR58671SD1Ev
    // CHECK-NEXT: ret void
    (f)(S(), 0);
  }

  void h() {
    g2(17);
  }
}

// PR6199
namespace PR6199 {
  struct A { ~A(); };

  struct B { operator A(); };

  // CHECK: define weak_odr void @_ZN6PR61992f2IiEENS_1AET_
  template<typename T> A f2(T) {
    B b;
    // CHECK: call void @_ZN6PR61991BcvNS_1AEEv
    // CHECK-NEXT: ret void
    return b;
  }

  template A f2<int>(int);
  
}

namespace T12 {

struct A { 
  A(); 
  ~A();
  int f();
};

int& f(int);

// CHECK: define void @_ZN3T121gEv
void g() {
  // CHECK: call void @_ZN3T121AC1Ev
  // CHECK-NEXT: call i32 @_ZN3T121A1fEv(
  // CHECK-NEXT: call i32* @_ZN3T121fEi(
  // CHECK-NEXT: call void @_ZN3T121AD1Ev(
  int& i = f(A().f());
}

}

namespace PR6648 {
  struct B {
    ~B();
  };
  B foo;
  struct D;
  D& zed(B);
  void foobar() {
    // CHECK: call %"struct.PR6648::D"* @_ZN6PR66483zedENS_1BE
    zed(foo);
  }
}
