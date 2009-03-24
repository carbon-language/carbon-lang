// RUN: clang-cc -fsyntax-only -verify %s

static void test() {
  int *pi;
  int x;
  typeof pi[x] y; 
}
