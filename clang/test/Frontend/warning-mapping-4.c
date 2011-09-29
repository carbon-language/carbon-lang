// RUN: %clang_cc1 -verify -Wno-error=sign-compare %s
// RUN: %clang_cc1 -verify -Wsign-compare -w -Wno-error=sign-compare %s

int f0(int x, unsigned y) {
  return x < y;
}
