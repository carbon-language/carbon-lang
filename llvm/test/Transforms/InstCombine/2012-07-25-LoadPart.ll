; RUN: opt < %s -instcombine -S | FileCheck %s
; PR13442

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"

@test = constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]

define i64 @foo() {
  %ret = load i64* bitcast (i8* getelementptr (i8* bitcast ([4 x i32]* @test to i8*), i64 2) to i64*), align 1
  ret i64 %ret
  ; CHECK: ret i64 844424930263040
}
