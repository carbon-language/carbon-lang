// RUN: %clang_cc1 -fsyntax-only -verify %s

void *test1(void) { return 0; }

void test2 (const struct {int a;} *x) {
  // expected-note@-1 {{variable 'x' declared const here}}

  x->a = 10;
  // expected-error-re@-1 {{cannot assign to variable 'x' with const-qualified type 'const struct (anonymous struct at {{.*}}assign.c:5:19) *'}}
}

typedef int arr[10];
void test3() {
  const arr b;
  const int b2[10]; 
  b[4] = 1; // expected-error {{read-only variable is not assignable}}
  b2[4] = 1; // expected-error {{read-only variable is not assignable}}
}
