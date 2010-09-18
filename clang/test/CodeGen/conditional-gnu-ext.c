// RUN: %clang_cc1 -emit-llvm %s -o %t
// PR1824

int foo(int x, short y) {
  return x ?: y;
}

// rdar://6586493
float test(float x, int Y) {
  return Y != 0 ? : x;
}

// rdar://8446940
extern void abort();
void  test1 () {
  char x[1];
  char *y = x ? : 0;

  if (x != y)
    abort();
}
