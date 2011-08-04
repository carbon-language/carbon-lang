// RUN: %clang_cc1 -fsyntax-only -verify %s

int main() {
  []{}; // expected-error {{expected expression}}
}
