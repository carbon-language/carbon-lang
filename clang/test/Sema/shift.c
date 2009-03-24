// RUN: clang-cc -fsyntax-only %s

void test() {
  char c;
  c <<= 14;
}
