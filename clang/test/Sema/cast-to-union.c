// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

union u { int i; };
void f(union u);

void test(int x) {
  f((union u)x); // expected-warning {{C99 forbids casts to union type}}
  f((union u)&x); // expected-error {{cast to union type from type 'int *' not present in union}}
}

union u w = (union u)2; // expected-warning {{C99 forbids casts to union type}}
union u ww = (union u)1.0; // expected-error{{cast to union type from type 'double' not present in union}}
union u x = 7; // expected-error{{incompatible type initializing 'int', expected 'union u'}}
int i;
union u zz = (union u)i; // expected-error{{initializer element is not a compile-time constant}}  expected-warning {{C99 forbids casts to union type}}

struct s {int a, b;};
struct s y = { 1, 5 };
struct s z = (struct s){ 1, 5 };
