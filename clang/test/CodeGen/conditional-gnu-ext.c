// RUN: clang-cc -emit-llvm %s -o %t
// PR1824

int foo(int x, short y) {
  return x ?: y;
}

// rdar://6586493
float test(float x, int Y) {
  return Y != 0 ? : x;
}

