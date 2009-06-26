// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x
void f() {
  auto a = a; // expected-error{{variable 'a' declared with 'auto' type cannot appear in its own initializer}}
}
