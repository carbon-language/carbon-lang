// RUN: %clang_cc1 -fsyntax-only -verify %s

void f() const; // expected-error {{type qualifier is not allowed on this function}}

typedef void cfn() const; 
cfn f2; // expected-error {{a qualified function type cannot be used to declare a nonmember function or a static member function}}

class C {
  void f() const;
  cfn f2;
  static void f3() const; // expected-error {{type qualifier is not allowed on this function}}
  static cfn f4; // expected-error {{a qualified function type cannot be used to declare a nonmember function or a static member function}}

  void m1() {
    x = 0;
  }

  void m2() const {
    x = 0; // expected-error {{read-only variable is not assignable}}
  }

  int x;
};
