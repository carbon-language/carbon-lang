// RUN: clang-cc -fsyntax-only -verify %s -fblocks

void tovoid(void*);

void tovoid_test(int (^f)(int, int)) {
  tovoid(f);
}
