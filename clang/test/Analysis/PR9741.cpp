// RUN: %clang_cc1 -cc1 -std=c++0x -Wuninitialized -verify %s

void f() {
  int a[] = { 1, 2, 3 };
  unsigned int u = 0;
  for (auto x : a)
    ;
}
