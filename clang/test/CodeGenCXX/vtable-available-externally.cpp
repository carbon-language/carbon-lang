// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -O3 -o %t 
// RUN: FileCheck --check-prefix=CHECK-TEST1 %s < %t
// RUN: FileCheck --check-prefix=CHECK-TEST2 %s < %t

#include <typeinfo>

// Test1::A's key function (f) is not defined in this translation unit, but in
// order to devirtualize calls, we emit the class related data with
// available_externally linkage.

// CHECK-TEST1: @_ZTVN5Test11AE = available_externally
// CHECK-TEST1: @_ZTSN5Test11AE = available_externally
// CHECK-TEST1: @_ZTIN5Test11AE = available_externally
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

// CHECK: define void @_ZN5Test11gEv
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

// PR9114, test that we don't try to emit a definition of 
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
