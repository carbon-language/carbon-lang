// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: = type { i32, [4 x i8] }
union Test1 {
  int a;
  int b: 39;
};

Test1 t1;
