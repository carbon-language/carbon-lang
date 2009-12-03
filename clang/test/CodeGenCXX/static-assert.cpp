// RUN: clang-cc %s -emit-llvm -o - -std=c++0x -verify

static_assert(true, "");

void f() {
  static_assert(true, "");
}
