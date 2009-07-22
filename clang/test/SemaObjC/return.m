// RUN: clang-cc %s -fsyntax-only -verify

int test1() {
  id a;
  @throw a;
}
