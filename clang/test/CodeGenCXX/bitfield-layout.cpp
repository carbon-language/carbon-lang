// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: %union.Test1 = type { i32, [4 x i8] }
union Test1 {
  int a;
  int b: 39;
} t1;

// CHECK: %union.Test2 = type { i8 }
union Test2 {
  int : 6;
} t2;

// CHECK: %union.Test3 = type { [2 x i8] }
union Test3 {
  int : 9;
} t3;
