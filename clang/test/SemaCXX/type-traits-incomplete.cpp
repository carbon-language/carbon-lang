// RUN: clang-cc -fsyntax-only -verify %s 

struct S; // expected-note{{forward declaration of 'struct S'}}

void f() {
  __is_pod(S); // expected-error{{incomplete type 'struct S' used in type trait expression}}
}
