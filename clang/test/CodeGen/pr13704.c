// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
extern void foo(__int128);

void bar() {
  __int128 x = 2;
  x--;
  foo(x);
// CHECK: add nsw i128 %0, -1
}
