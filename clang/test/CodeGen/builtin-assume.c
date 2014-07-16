// RUN: %clang_cc1 -triple i386-mingw32 -fms-extensions -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: @test1
int test1(int *a) {
  __assume(a != 0);
  return a[0];
}

