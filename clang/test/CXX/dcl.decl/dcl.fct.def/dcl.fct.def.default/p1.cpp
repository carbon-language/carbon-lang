// RUN: %clang_cc1 -verify %s -std=c++11

// A function that is explicitly defaulted shall
struct A {
  // -- be a special member function,
  A(int) = default; // expected-error {{only special member functions may be defaulted}}

  // -- have the same declared function type as if it had been implicitly
  //    declared
  void operator=(const A &) = default; // expected-error {{must return 'A &'}}
  A(...) = default; // expected-error {{cannot be variadic}}
  A(const A &, ...) = default; // expected-error {{cannot be variadic}}

  //    (except for possibly differing ref-qualifiers
  A &operator=(A &&) & = default;

  //    and except that in the case of a copy constructor or copy assignment
  //    operator, the parameter type may be "reference to non-const T")
  A(A &) = default;
  A &operator=(A &) = default;

  // -- not have default arguments
  A(double = 0.0) = default; // expected-error {{cannot have default arguments}}
  A(const A & = 0) = default; // expected-error {{cannot have default arguments}}
};
