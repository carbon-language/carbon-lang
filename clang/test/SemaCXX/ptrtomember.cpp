// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

struct  S {
	int i;

	int mem(int);
};

int foo(int S::* ps, S *s)
{
    return (s->*ps)(1); // expected-error {{called object type 'int' is not a function or function pointer}}
}

struct S2 {
  int bitfield : 1;
};

int S2::*pf = &S2::bitfield; // expected-error {{address of bit-field requested}}
