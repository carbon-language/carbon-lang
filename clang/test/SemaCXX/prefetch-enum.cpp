// RUN: %clang_cc1 -fsyntax-only %s -verify
// PR5679

enum X { A = 3 };

void Test() {
  char ch;
  __builtin_prefetch(&ch, 0, A);
}
