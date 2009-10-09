// RUN: clang-cc -fsyntax-only -verify %s
struct A; // expected-note 3 {{forward declaration of 'struct A'}}

A f(); // expected-note {{note: 'f' declared here}}

struct B {
};

void g() {
  f(); // expected-error {{calling 'f' with incomplete return type 'struct A'}}

  typedef A (*Func)();
  Func fp;
  fp(); // expected-error {{calling function with incomplete return type 'struct A'}}
  ((Func)0)();  // expected-error {{calling function with incomplete return type 'struct A'}}  
}
