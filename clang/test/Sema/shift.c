// RUN: clang -fsyntax-only %s

void test() {
  char c;
  c <<= 14;
}
