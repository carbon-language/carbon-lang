// RUN: %clang_cc1 -emit-llvm-only -std=c++11 -g %s

namespace PR9414 {
  int f() {
    auto x = 0;
    return x;
  }
}
