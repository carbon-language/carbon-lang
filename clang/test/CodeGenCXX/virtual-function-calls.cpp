// RUN: clang-cc -emit-llvm-only %s

// PR5021
struct A {
  virtual void f(char);
};

void f(A *a) {
  a->f('c');
}
