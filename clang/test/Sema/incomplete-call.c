// RUN: %clang_cc1 -fsyntax-only -verify %s

struct foo; // expected-note 3 {{forward declaration of 'struct foo'}}

struct foo a(); // expected-note {{'a' declared here}}
void b(struct foo);
void c();

void func() {
  a(); // expected-error{{calling 'a' with incomplete return type 'struct foo'}}
  b(*(struct foo*)0); // expected-error{{argument type 'struct foo' is incomplete}}
  c(*(struct foo*)0); // expected-error{{argument type 'struct foo' is incomplete}}
}
