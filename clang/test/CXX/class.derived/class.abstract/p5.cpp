// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A {
  virtual void f() = 0; // expected-note{{pure virtual function}}
};

struct B : A {
  virtual void f();
};

struct C : B {
  virtual void f() = 0; // expected-note 2{{pure virtual function}}
};

struct D : C {
};

void test() {
  (void)new A; // expected-error{{object of abstract type}}
  (void)new B;
  (void)new C; // expected-error{{object of abstract type}}
  (void)new D; // expected-error{{object of abstract type}}
}
