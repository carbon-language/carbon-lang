// RUN: clang-cc -fsyntax-only %s -verify

typedef struct { unsigned long bits[(((1) + (64) - 1) / (64))]; } cpumask_t;
cpumask_t x;
void foo() {
  (void)x;
}

