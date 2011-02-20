// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++0x

struct X {};
typedef X foo_t;

foo_t *ptr;
char c1 = ptr; // expected-error{{'foo_t *' (aka 'X *')}}

const foo_t &ref = foo_t();
char c2 = ref; // expected-error{{'const foo_t' (aka 'const X')}}

// deduced auto should not produce an aka.
auto aut = X();
char c3 = aut; // expected-error{{from 'X' to 'char'}}
