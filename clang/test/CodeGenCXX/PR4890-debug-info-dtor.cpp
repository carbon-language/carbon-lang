// RUN: clang-cc -emit-llvm-only -g %s
struct X {
  ~X();
};

X::~X() { }
