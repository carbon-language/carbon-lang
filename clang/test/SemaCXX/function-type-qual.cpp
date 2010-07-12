// RUN: %clang_cc1 -fsyntax-only -verify %s

void f() const; // expected-error {{type qualifier is not allowed on this function}}
void (*pf)() const; // expected-error {{type qualifier is not allowed on this function pointer}}
void (&rf)() const = f; // expected-error {{type qualifier is not allowed on this function reference}}

typedef void cfn() const; 
cfn f2; // expected-error {{a qualified function type cannot be used to declare a nonmember function}}

class C {
  void f() const;
  cfn f2;
  static void f3() const; // expected-error {{type qualifier is not allowed on this function}}
  static cfn f4; // expected-error {{a qualified function type cannot be used to declare a static member function}}

  void m1() {
    x = 0;
  }

  void m2() const {
    x = 0; // expected-error {{read-only variable is not assignable}}
  }

  int x;
};

void (C::*mpf)() const;
cfn C::*mpg;
