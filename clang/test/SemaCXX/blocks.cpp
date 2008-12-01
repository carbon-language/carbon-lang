// RUN: clang -fsyntax-only -verify %s

void tovoid(void*);

void tovoid_test(int (^f)(int, int)) {
  tovoid(f);
}
