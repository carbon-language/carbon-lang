// RUN: clang -fsyntax-only -verify %s

class A {
  virtual void f();
  virtual void g() = 0;

  void h() = 0; // expected-error {{'h' is not virtual and cannot be declared pure}}
  void i() = 1; // expected-error {{initializer on function does not look like a pure-specifier}}
  void j() = 0u; // expected-error {{initializer on function does not look like a pure-specifier}}

public:
  A(int);
};

class B : public A {
  // Needs to recognize that overridden function is virtual.
  //void g() = 0;

  // Needs to recognize that function does not override.
  //void g(int) = 0;
};

// Needs to recognize invalid uses of abstract classes.
/*
A fn(A)
{
  A a;
  static_cast<A>(0);
  try {
  } catch(A) {
  }
}
*/
