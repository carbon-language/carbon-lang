// RUN: clang -emit-llvm %s -o -
void f1() {
  // Scalars in braces.
  int a = { 1 };
  int b = { 1, 2 };
}

void f2() {
  int a[2][2] = { { 1, 2 }, { 3, 4 } };
  int b[3][3] = { { 1, 2 }, { 3, 4 } };
  int *c[2] = { &a[1][1], &b[2][2] };
  int *d[2][2] = { {&a[1][1], &b[2][2]}, {&a[0][0], &b[1][1]} };
  int *e[3][3] = { {&a[1][1], &b[2][2]}, {&a[0][0], &b[1][1]} };
  char ext[3][3] = {".Y",".U",".V"};
}

typedef void (* F)(void);
extern void foo(void);
struct S { F f; };
void f3() {
  struct S a[1] = { { foo } };
}

