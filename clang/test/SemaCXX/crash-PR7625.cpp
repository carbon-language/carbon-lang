// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T> struct a : T {
 struct x : T {
   int aa() { return p; } // expected-error{{use of undeclared identifier 'p'}}
 };
};
