// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s
// expected-no-diagnostics
int main() {
  char *s;
  s = (char []){"whatever"};
  s = (char(*)){s};
}
