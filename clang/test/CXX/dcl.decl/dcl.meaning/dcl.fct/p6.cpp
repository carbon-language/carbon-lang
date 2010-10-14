// RUN: %clang_cc1 -fsyntax-only -verify %s

void f() const; // expected-error{{type qualifier is not allowed on this function}}

struct X {
  void f() const;
  friend void g() const; // expected-error{{type qualifier is not allowed on this function}}
  static void h() const; // expected-error{{type qualifier is not allowed on this function}}
};

struct Y {
  friend void X::f() const;
  friend void ::f() const; // expected-error{{type qualifier is not allowed on this function}}
};
