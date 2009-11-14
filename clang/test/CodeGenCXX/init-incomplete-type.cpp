// RUN: clang-cc %s -emit-llvm-only -verify
// PR5489

template<typename E>
struct Bar {
 int x_;
};

static struct Bar<int> bar[1] = {
  { 0 }
};

