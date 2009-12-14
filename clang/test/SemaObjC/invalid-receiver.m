// RUN: clang -cc1 -fsyntax-only -verify %s

typedef struct NotAClass {
  int a, b;
} NotAClass;

void foo() {
  [NotAClass nonexistent_method]; // expected-error {{invalid receiver to message expression}}
}
