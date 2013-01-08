// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s
// rdar://12655829

void f() {
  struct { int x; int y[]; } a; // expected-note {{'a' declared here}}
  ^{return a.x;}(); // expected-error {{cannot refer to declaration of structure variable with flexible array member inside block}}
}
