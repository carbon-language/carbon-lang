// RUN: clang -fsyntax-only -verify %s 

typedef int INT;

class Foo {
  Foo();
  (Foo)(float) { }
  explicit Foo(int); // expected-error{{previous declaration is here}}
  Foo(const Foo&);

  ((Foo))(INT); // expected-error{{cannot be redeclared}}

  Foo(Foo foo, int i = 17, int j = 42); // expected-error {{copy constructor must pass its first argument by reference}}

  static Foo(short, short); // expected-error{{constructor cannot be declared 'static'}}
  virtual Foo(double); // expected-error{{constructor cannot be declared 'virtual'}}
  Foo(long) const; // expected-error{{'const' qualifier is not allowed on a constructor}}

  int Foo(int, int); // expected-error{{constructor cannot have a return type}}
};
