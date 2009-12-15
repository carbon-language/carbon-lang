// RUN: %clang_cc1 -emit-llvm-only -g %s
struct X {
  ~X();
};

X::~X() { }
