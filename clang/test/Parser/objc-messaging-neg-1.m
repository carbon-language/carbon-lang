// RUN: %clang_cc1 -fsyntax-only -verify %s

int main() {
  id a;
  [a bla:0 6:7]; // expected-error {{expected ']'}}
}
