// RUN: clang-cc -fsyntax-only -verify %s
struct T { 
  void f();
};

struct A {
  T* operator->(); // expected-note{{candidate function}}
};

struct B {
  T* operator->(); // expected-note{{candidate function}}
};

struct C : A, B {
};

struct D : A { };

void f(C &c, D& d) {
  c->f(); // expected-error{{use of overloaded operator '->' is ambiguous}}
  d->f();
}