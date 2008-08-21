// RUN: clang -emit-llvm %s -o %t
// PR1824

int foo(int x, short y) {
  return x ?: y;
}
