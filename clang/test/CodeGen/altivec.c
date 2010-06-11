// RUN: %clang_cc1 -faltivec -triple powerpc-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// CHECK: @test0 = global <4 x i32> <i32 1, i32 1, i32 1, i32 1>
vector int test0 = (vector int)(1);
