// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

struct X1 {
  X1();
};

struct X2 {
  X2();
  ~X2();
};

void vararg(...);

void f(X1 x1, X2 x2) {
  vararg(x1); // okay
  vararg(x2); // expected-error{{cannot pass object of non-trivial type 'X2' through variadic function; call will abort at runtime}}
}
