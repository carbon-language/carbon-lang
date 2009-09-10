// RUN: clang-cc -fsyntax-only -verify %s

// C++-specific tests for integral constant expressions.

const int c = 10;
int ar[c];

struct X0 {
  static const int value = static_cast<int>(4.0);
};

void f() {
  if (const int value = 17) {
    int array[value];
  }
}
