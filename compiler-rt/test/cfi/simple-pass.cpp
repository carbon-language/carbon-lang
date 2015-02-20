// RUN: %clangxx_cfi -o %t %s
// RUN: %t

// Tests that the CFI mechanism does not crash the program when making various
// kinds of valid calls involving classes with various different linkages and
// types of inheritance.

inline void break_optimization(void *arg) {
  __asm__ __volatile__("" : : "r" (arg) : "memory");
}

struct A {
  virtual void f();
};

void A::f() {}

struct A2 : A {
  virtual void f();
};

void A2::f() {}

struct B {
  virtual void f() {}
};

struct B2 : B {
  virtual void f() {}
};

namespace {

struct C {
  virtual void f();
};

void C::f() {}

struct C2 : C {
  virtual void f();
};

void C2::f() {}

struct D {
  virtual void f() {}
};

struct D2 : D {
  virtual void f() {}
};

}

struct E {
  virtual void f() {}
};

struct E2 : virtual E {
  virtual void f() {}
};

int main() {
  A *a = new A;
  break_optimization(a);
  a->f();
  a = new A2;
  break_optimization(a);
  a->f();

  B *b = new B;
  break_optimization(b);
  b->f();
  b = new B2;
  break_optimization(b);
  b->f();

  C *c = new C;
  break_optimization(c);
  c->f();
  c = new C2;
  break_optimization(c);
  c->f();

  D *d = new D;
  break_optimization(d);
  d->f();
  d = new D2;
  break_optimization(d);
  d->f();

  E *e = new E;
  break_optimization(e);
  e->f();
  e = new E2;
  break_optimization(e);
  e->f();
}
