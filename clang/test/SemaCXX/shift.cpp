// RUN: %clang_cc1 -Wall -Wshift-sign-overflow -ffreestanding -fsyntax-only -verify %s

#include <limits.h>

#define WORD_BIT (sizeof(int) * CHAR_BIT)

template <int N> void f() {
  (void)(N << 30); // expected-warning {{the promoted type of the shift expression is 'int'}}
  (void)(30 << N); // expected-warning {{the promoted type of the shift expression is 'int'}}
}

void test() {
  f<30>(); // expected-note {{instantiation}}
}
