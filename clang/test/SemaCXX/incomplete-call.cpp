// RUN: clang-cc -fsyntax-only -verify %s
struct A; // expected-note 8 {{forward declaration of 'struct A'}}

A f(); // expected-note {{note: 'f' declared here}}

struct B {
  A f(); // expected-note {{'f' declared here}}
  A operator()(); // expected-note {{'operator()' declared here}}
  operator A(); // expected-note {{'operator A' declared here}}
  A operator!(); // expected-note 2 {{'operator!' declared here}}
};

void g() {
  f(); // expected-error {{calling 'f' with incomplete return type 'struct A'}}

  typedef A (*Func)();
  Func fp;
  fp(); // expected-error {{calling function with incomplete return type 'struct A'}}
  ((Func)0)();  // expected-error {{calling function with incomplete return type 'struct A'}}  
  
  B b;
  b.f(); // expected-error {{calling 'f' with incomplete return type 'struct A'}}
  
  b.operator()(); // expected-error {{calling 'operator()' with incomplete return type 'struct A'}}
  b.operator A(); // expected-error {{calling 'operator A' with incomplete return type 'struct A'}}
  b.operator!(); // expected-error {{calling 'operator!' with incomplete return type 'struct A'}}
  
  !b; // expected-error {{calling 'operator!' with incomplete return type 'struct A'}}
}
