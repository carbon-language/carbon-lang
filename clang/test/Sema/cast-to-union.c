// RUN: clang -fsyntax-only -verify -pedantic %s

union u { int i; };
void f(union u);

void test(int x) {
  f((union u)x); // expected-warning {{C99 forbids casts to union type}}
  f((union u)&x); // expected-error {{cast to union type from type 'int *' not present in union}}
}
