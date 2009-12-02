// RUN: clang-cc %s -emit-llvm-only

struct A { virtual ~A(); };
struct B : A {
  ~B() { }
};
B x;

