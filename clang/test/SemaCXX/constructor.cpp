// RUN: clang -fsyntax-only -verify %s 

class Foo {
  Foo();
  (Foo)(float) { }
  explicit Foo(int);
  Foo(const Foo&);

  static Foo(short, short); // expected-error{{constructor cannot be declared 'static'}}
  virtual Foo(double); // expected-error{{constructor cannot be declared 'virtual'}}
  Foo(long) const; // expected-error{{'const' qualifier is not allowed on a constructor}}

  int Foo(int, int); // expected-error{{constructor cannot have a return type}}
};
