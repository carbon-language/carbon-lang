// RUN: clang-cc -emit-llvm %s -o %t

int f() {
 int a[2];
 a[0] = 0;
}

int f2() {
  int x = 0;
  int y = 1;
  int a[10] = { y, x, 2, 3};
  int b[10] = { 2,4,x,6,y,8};
  int c[5] = { 0,1,2,3};
}
