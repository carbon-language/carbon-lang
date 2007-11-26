// RUN: clang -emit-llvm %s
// PR1824

int foo(int x, short y) {
  return x ?: y;
}
