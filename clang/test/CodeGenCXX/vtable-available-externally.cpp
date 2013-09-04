// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -o %t 
// RUN: FileCheck --check-prefix=CHECK-TEST1 %s < %t
// RUN: FileCheck --check-prefix=CHECK-TEST2 %s < %t
// RUN: FileCheck --check-prefix=CHECK-TEST5 %s < %t

#include <typeinfo>

// CHECK-TEST1: @_ZTVN5Test11AE = external unnamed_addr constant
namespace Test1 {

struct A {
  A();
  virtual void f();
  virtual ~A() { }
};

A::A() { }

void f(A* a) {
  a->f();
};

// CHECK-LABEL: define void @_ZN5Test11gEv
// CHECK: call void @_ZN5Test11A1fEv
void g() {
  A a;
  f(&a);
}

}

// Test2::A's key function (f) is defined in this translation unit, but when
// we're doing codegen for the typeid(A) call, we don't know that yet.
// This tests mainly that the typeinfo and typename constants have their linkage
// updated correctly.

// CHECK-TEST2: @_ZTSN5Test21AE = constant
// CHECK-TEST2: @_ZTIN5Test21AE = unnamed_addr constant
// CHECK-TEST2: @_ZTVN5Test21AE = unnamed_addr constant
namespace Test2 {
  struct A {
    virtual void f();
  };

  const std::type_info &g() {
    return typeid(A);
  };

  void A::f() { }
}

// Test that we don't assert on this test.
namespace Test3 {

struct A {
  virtual void f();
  virtual ~A() { }
};

struct B : A {
  B();
  virtual void f();
};

B::B() { }

void g(A* a) {
  a->f();
};

}

// PR9114, test that we don't try to instantiate RefPtr<Node>.
namespace Test4 {

template <class T> struct RefPtr {
  T* p;
  ~RefPtr() {
    p->deref();
  }
};

struct A {
  virtual ~A();
};

struct Node;

struct B : A {
  virtual void deref();
  RefPtr<Node> m;
};

void f() {
  RefPtr<B> b;
}

}

// PR9130, test that we emit a definition of A::f.
// CHECK-TEST5-LABEL: define linkonce_odr void @_ZN5Test51A1fEv
namespace Test5 {

struct A {
  virtual void f() { }
};

struct B : A { 
  virtual ~B();
};

B::~B() { }

}

// Check that we don't assert on this test.
namespace Test6 {

struct A {
  virtual ~A();
  int a;
};

struct B {
  virtual ~B();
  int b;
};

struct C : A, B { 
  C();
};

struct D : C {
  virtual void f();
  D();
};

D::D() { }

}

namespace Test7 {

struct c1 {};
struct c10 : c1{
  virtual void foo ();
};
struct c11 : c10, c1{
  virtual void f6 ();
};
struct c28 : virtual c11{
  void f6 ();
};
}
