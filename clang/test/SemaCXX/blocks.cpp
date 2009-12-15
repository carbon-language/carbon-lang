// RUN: %clang_cc1 -fsyntax-only -verify %s -fblocks

void tovoid(void*);

void tovoid_test(int (^f)(int, int)) {
  tovoid(f);
}

void reference_lvalue_test(int& (^f)()) {
  f() = 10;
}
