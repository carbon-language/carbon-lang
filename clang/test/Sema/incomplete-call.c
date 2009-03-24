// RUN: clang-cc -fsyntax-only -verify %s

struct foo; // expected-note 3 {{forward declaration of 'struct foo'}}

struct foo a();
void b(struct foo);
void c();

void func() {
  a(); // expected-error{{return type of called function ('struct foo') is incomplete}}
  b(*(struct foo*)0); // expected-error{{argument type 'struct foo' is incomplete}}
  c(*(struct foo*)0); // expected-error{{argument type 'struct foo' is incomplete}}
}
