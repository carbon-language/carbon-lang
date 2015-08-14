; RUN: opt < %s -basicaa -dse -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@A = common global [100 x i32] zeroinitializer, align 16
@x = common global i32 0

; Negative Test case-
;void foo(int N) {
;  A[0] = N;
;  for(int i=0;i<N;++i)
;    A[i]+=i;
;  A[0] = 10;
;}
;; Stores should not be optimized away.

define void @test_01(i32 %N) #0 {
  %1 = alloca i32
  %i = alloca i32
  store i32 %N, i32* %1
  %2 = load i32, i32* %1
  store i32 %2, i32* getelementptr inbounds ([100 x i32], [100 x i32]* @A, i32 0, i64 0)
  store i32 0, i32* %i
  br label %3

; <label>:3                                       ; preds = %14, %0
  %4 = load i32, i32* %i
  %5 = load i32, i32* %1
  %6 = icmp slt i32 %4, %5
  br i1 %6, label %7, label %17

; <label>:7                                       ; preds = %3
  %8 = load i32, i32* %i
  %9 = load i32, i32* %i
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds [100 x i32], [100 x i32]* @A, i32 0, i64 %10
  %12 = load i32, i32* %11
  %13 = add nsw i32 %12, %8
  store i32 %13, i32* %11
  br label %14

; <label>:14                                      ; preds = %7
  %15 = load i32, i32* %i
  %16 = add nsw i32 %15, 1
  store i32 %16, i32* %i
  br label %3

; <label>:17                                      ; preds = %3
  store i32 10, i32* getelementptr inbounds ([100 x i32], [100 x i32]* @A, i32 0, i64 0)
  ret void
}
; CHECK-LABEL: @test_01(
; CHECK: store i32 %2, i32* getelementptr inbounds ([100 x i32], [100 x i32]* @A, i32 0, i64 0)
; CHECK: store i32 %13, i32* %11
; CHECK: store i32 10, i32* getelementptr inbounds ([100 x i32], [100 x i32]* @A, i32 0, i64 0)


; Postive Test case-
;void foo(int N) {
;  A[0] = N;
;  for(int i=0;i<N;++i)
;    A[i]=i;
;  A[0] = 10;
;}
;; Stores should not be optimized away.
define void @test_02(i32 %N) #0 {
  %1 = alloca i32
  %i = alloca i32
  store i32 %N, i32* %1
  %2 = load i32, i32* %1
  store i32 %2, i32* getelementptr inbounds ([100 x i32], [100 x i32]* @A, i32 0, i64 0)
  store i32 0, i32* %i
  br label %3

; <label>:3                                       ; preds = %12, %0
  %4 = load i32, i32* %i
  %5 = load i32, i32* %1
  %6 = icmp slt i32 %4, %5
  br i1 %6, label %7, label %15

; <label>:7                                       ; preds = %3
  %8 = load i32, i32* %i
  %9 = load i32, i32* %i
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds [100 x i32], [100 x i32]* @A, i32 0, i64 %10
  store i32 %8, i32* %11
  br label %12

; <label>:12                                      ; preds = %7
  %13 = load i32, i32* %i
  %14 = add nsw i32 %13, 1
  store i32 %14, i32* %i
  br label %3

; <label>:15                                      ; preds = %3
  store i32 10, i32* getelementptr inbounds ([100 x i32], [100 x i32]* @A, i32 0, i64 0)
  ret void
}

; CHECK-LABEL: @test_02(
; CHECK-NOT: store i32 %2, i32* getelementptr inbounds ([100 x i32], [100 x i32]* @A, i32 0, i64 0)
; CHECK: store i32 %7, i32* %10
; CHECK: store i32 10, i32* getelementptr inbounds ([100 x i32], [100 x i32]* @A, i32 0, i64 0)


