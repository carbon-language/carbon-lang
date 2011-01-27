// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

void f0() &; // expected-error{{ref-qualifier '&' is only allowed on non-static member functions, member function pointers, and typedefs of function types}}
void f1() &&; // expected-error{{ref-qualifier '&&' is only allowed on non-static member functions, member function pointers, and typedefs of function types}}

struct X {
  void f0() &; 
  void f1() &&;
  static void f2() &; // expected-error{{ref-qualifier '&' is only allowed on non-static member functions, member function pointers, and typedefs of function types}}
  static void f3() &&; // expected-error{{ref-qualifier '&&' is only allowed on non-static member functions, member function pointers, and typedefs of function types}}
};

typedef void func_type_lvalue() &;
typedef void func_type_rvalue() &&;

func_type_lvalue f2; // expected-error{{nonmember function cannot have a ref-qualifier '&'}}
func_type_rvalue f3; // expected-error{{nonmember function cannot have a ref-qualifier '&&'}}

struct Y {
  func_type_lvalue f0;
  func_type_rvalue f1;
};

void (X::*mpf1)() & = &X::f0;
void (X::*mpf2)() && = &X::f1;


void (f() &&); // expected-error{{ref-qualifier '&&' is only allowed on non-static member functions, member function pointers, and typedefs of function types}}
