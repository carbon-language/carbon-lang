// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

void test0() {
  // CHECK: define void @test0()
  // CHECK:      [[F:%.*]] = alloca float
  // CHECK-NEXT: [[REAL:%.*]] = volatile load float* getelementptr inbounds ({{%.*}} @test0_v, i32 0, i32 0)
  // CHECK-NEXT: volatile load float* getelementptr inbounds ({{%.*}} @test0_v, i32 0, i32 1)
  // CHECK-NEXT: store float [[REAL]], float* [[F]], align 4
  // CHECK-NEXT: ret void
  extern volatile _Complex float test0_v;
  float f = (float) test0_v;
}

void test1() {
  // CHECK: define void @test1()
  // CHECK:      [[REAL:%.*]] = volatile load float* getelementptr inbounds ({{%.*}} @test1_v, i32 0, i32 0)
  // CHECK-NEXT: [[IMAG:%.*]] = volatile load float* getelementptr inbounds ({{%.*}} @test1_v, i32 0, i32 1)
  // CHECK-NEXT: volatile store float [[REAL]], float* getelementptr inbounds ({{%.*}} @test1_v, i32 0, i32 0)
  // CHECK-NEXT: volatile store float [[IMAG]], float* getelementptr inbounds ({{%.*}} @test1_v, i32 0, i32 1)
  // CHECK-NEXT: ret void
  extern volatile _Complex float test1_v;
  test1_v = test1_v;
}
