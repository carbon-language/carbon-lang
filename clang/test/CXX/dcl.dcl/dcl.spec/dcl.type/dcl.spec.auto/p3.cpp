// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x
void f() {
  auto a = a; // expected-error{{variable 'a' declared with 'auto' type cannot appear in its own initializer}}
}

void g() {
  auto a; // expected-error{{declaration of variable 'a' with type 'auto' requires an initializer}}
  
  auto *b; // expected-error{{declaration of variable 'b' with type 'auto *' requires an initializer}}
}
