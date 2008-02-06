// RUN: clang -fsyntax-only -verify %s

void *test1(void) { return 0; }

void test2 (const struct {int a;} *x) {
  x->a = 10; // expected-error {{read-only variable is not assignable}}
}
