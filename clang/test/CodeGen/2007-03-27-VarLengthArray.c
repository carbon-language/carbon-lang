// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

// CHECK: getelementptr inbounds i32* %vla
extern void f(int *);
int e(int m, int n) {
  int x[n];
  f(x);
  return x[m];
}
