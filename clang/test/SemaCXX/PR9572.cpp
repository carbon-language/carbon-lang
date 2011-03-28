// RUN: %clang_cc1 -fsyntax-only -verify %s
class Base {
  virtual ~Base();
};
struct Foo : public Base {
  const int kBlah = 3; // expected-error{{fields can only be initialized in constructors}}
  Foo();
};
struct Bar : public Foo {
  Bar() { }
};
struct Baz {
  Foo f;
  Baz() { }
};
