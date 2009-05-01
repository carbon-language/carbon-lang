// RUN: clang-cc -fsyntax-only %s -verify

typedef struct { unsigned long bits[(((1) + (64) - 1) / (64))]; } cpumask_t;
cpumask_t x;
void foo() {
  (void)x;
}
void bar() {
  char* a;
  double b;
  b = (double)a; // expected-error {{pointer cannot be cast to type}}
  a = (char*)b; // expected-error {{cannot be cast to a pointer type}}
}

