// RUN: clang-cc -emit-llvm < %s | FileCheck %s

// CHECK: @test1.x = internal constant [12 x i32] [i32 1
// CHECK: @test2.x = internal constant [13 x i32] [i32 1,

#include <string.h>

void test1() {
  // This should codegen as a "@test1.x" global.
  const int x[] = { 1, 2, 3, 4, 6, 8, 9, 10, 123, 231, 123,23 };
  foo(x);

// CHECK: @test1()
// CHECK: {{call.*@foo.*@test1.x}}
}


// rdar://7346691
void test2() {
  // This should codegen as a "@test2.x" global + memcpy.
  int x[] = { 1, 2, 3, 4, 6, 8, 9, 10, 123, 231, 123,23, 24 };
  foo(x);
  
  // CHECK: @test2()
  // CHECK: %x = alloca [13 x i32]
  // CHECK: call void @llvm.memcpy
  // CHECK: call{{.*}}@foo{{.*}}i32* %
}


void test3() {
  // This should codegen as a "@test3.x" global + memcpy.
  int x[100] = { 0 };
  foo(x);
  
  // CHECK: @test3()
  // CHECK: %x = alloca [100 x i32]
  // CHECK: call void @llvm.memset
}

