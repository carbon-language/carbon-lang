// RUN: %clang_cc1 %s -fsyntax-only -verify -Wweak-vtables

struct A { // expected-warning {{'A' has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit}}
  virtual void f() { } 
};

template<typename T> struct B {
  virtual void f() { } 
};

namespace {
  struct C { 
    virtual void f() { }
  };
}

void f() {
  struct A {
    virtual void f() { }
  };

  A *a;
  a->f();
}

// Use the vtables
void uses(A &a, B<int> &b, C &c) {
  a.f();
  b.f();
  c.f();
}
