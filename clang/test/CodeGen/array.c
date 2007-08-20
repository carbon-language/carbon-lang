// RUN: clang -emit-llvm %s

int f() {
 int a[2];
 a[0] = 0;
}
