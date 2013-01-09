// RUN: %clang_cc1 -fsyntax-only -fblocks -verify -std=c++11 %s
// rdar://12655829

void f() {
  struct { int x; int y[]; } a; // expected-note 2 {{'a' declared here}}
  ^{return a.x;}(); // expected-error {{cannot refer to declaration of structure variable with flexible array member inside block}}
  [] {return a.x;}(); // expected-error {{variable 'a' with flexible array member cannot be captured in a lambda expression}}
}
