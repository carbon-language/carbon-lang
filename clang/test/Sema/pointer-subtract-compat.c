// RUN: clang %s -fsyntax-only -verify -pedantic

typedef const char rchar;
int a(char* a, rchar* b) {
  return a-b;
}
