// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-apple-darwin9 -std=c++11 %s
// expected-no-diagnostics

#pragma ms_struct on

struct A {
  unsigned long a:4;
  unsigned char b;
  A();
};

struct B : public A {
  unsigned long c:16;
	int d;
  B();
};

static_assert(__builtin_offsetof(B, d) == 12,
  "We can't allocate the bitfield into the padding under ms_struct");