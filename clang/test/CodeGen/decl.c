// RUN: clang-cc -emit-llvm < %s | FileCheck %s

// CHECK: @test1.x = internal constant [12 x i32] [i32 1

void test1() {
  // This should codegen as a "@test1.x" global.
  const int x[] = { 1, 2, 3, 4, 6, 8, 9, 10, 123, 231, 123,23 };
  foo(x);

// CHECK: @test1()
// CHECK: {{call.*@foo.*@test1.x}}
}
