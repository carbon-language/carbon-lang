// RUN: %clang_cc1 -fsyntax-only -verify %s 

void choice(int);
int choice(bool);

void test() {
  // Result of ! must be type bool.
  int i = choice(!1);
}
