// RUN: clang-tidy -checks='-*,misc-undelegated-constructor' %s -- -std=c++11 2>&1 | FileCheck %s -implicit-check-not='{{warning:|error:}}'

struct Ctor;
Ctor foo();

struct Ctor {
  Ctor();
  Ctor(int);
  Ctor(int, int);
  Ctor(Ctor *i) {
    Ctor();
// CHECK: :[[@LINE-1]]:5: warning: did you intend to call a delegated constructor? A temporary object is created here instead
    Ctor(0);
// CHECK: :[[@LINE-1]]:5: warning: did you intend to call a delegated constructor? A temporary object is created here instead
    Ctor(1, 2);
// CHECK: :[[@LINE-1]]:5: warning: did you intend to call a delegated constructor? A temporary object is created here instead
    foo();
  }
};

Ctor::Ctor() {
  Ctor(1);
// CHECK: :[[@LINE-1]]:3: warning: did you intend to call a delegated constructor? A temporary object is created here instead
}

Ctor::Ctor(int i) : Ctor(i, 1) {} // properly delegated.

struct Dtor {
  Dtor();
  Dtor(int);
  Dtor(int, int);
  Dtor(Ctor *i) {
    Dtor();
// CHECK: :[[@LINE-1]]:5: warning: did you intend to call a delegated constructor? A temporary object is created here instead
    Dtor(0);
// CHECK: :[[@LINE-1]]:5: warning: did you intend to call a delegated constructor? A temporary object is created here instead
    Dtor(1, 2);
// CHECK: :[[@LINE-1]]:5: warning: did you intend to call a delegated constructor? A temporary object is created here instead
  }
  ~Dtor();
};

struct Base {};
struct Derived : public Base {
  Derived() { Base(); }
// CHECK: :[[@LINE-1]]:15: warning: did you intend to call a delegated constructor? A temporary object is created here instead
};

template <typename T>
struct TDerived : public Base {
  TDerived() { Base(); }
};

TDerived<int> t;
