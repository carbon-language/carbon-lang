// RUN: %clang_cc1 -fsyntax-only %s -verify

struct S {
 int one;
 int two;
};

struct S const foo(void);  // expected-warning{{type qualifier on return type has no effect}}


struct S tmp;

void priv_sock_init() {
  tmp = (struct S)foo();
}
