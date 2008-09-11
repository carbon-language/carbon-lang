// RUN: clang -fsyntax-only -verify %s 

void test() {
  int x;
  do
    int x;
  while (1);
}
