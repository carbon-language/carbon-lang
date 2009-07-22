// RUN: clang-cc %s -fsyntax-only -verify

int test1() {
  throw;
}
