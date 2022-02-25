; RUN: opt %loadPolly -polly-import-jscop \
; RUN: \
; RUN: -polly-codegen -S < %s | FileCheck %s
;
;    void foo(float A[], float B[]) {
;      for (long i = 0; i < 100; i++)
;        *(int *)(&A[i]) = *(int *)(&B[i]);
;      for (long i = 0; i < 100; i++)
;        A[i] += 10;
;    }

; CHECK: %polly.access.cast.A14 = bitcast float* %A to i32*
; CHECK: %[[R1:[._0-9]*]] = sub nsw i64 0, %polly.indvar11
; CHECK: %[[R2:[._0-9]*]] = add nsw i64 %[[R1]], 99
; CHECK: %polly.access.A15 = getelementptr i32, i32* %polly.access.cast.A14, i64 %[[R2]]
; CHECK: %[[R3:[._0-9]*]] = bitcast i32* %polly.access.A15 to float*
; CHECK: %tmp14_p_scalar_ = load float, float* %[[R3]], align 4, !alias.scope !3, !noalias !4

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @different_types(float* %A, float* %B) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb8, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp9, %bb8 ]
  %exitcond1 = icmp ne i64 %i.0, 100
  br i1 %exitcond1, label %bb3, label %bb10

bb3:                                              ; preds = %bb2
  %tmp = getelementptr inbounds float, float* %B, i64 %i.0
  %tmp4 = bitcast float* %tmp to i32*
  %tmp5 = load i32, i32* %tmp4, align 4
  %tmp6 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp7 = bitcast float* %tmp6 to i32*
  store i32 %tmp5, i32* %tmp7, align 4
  br label %bb8

bb8:                                              ; preds = %bb3
  %tmp9 = add nuw nsw i64 %i.0, 1
  br label %bb2

bb10:                                             ; preds = %bb2
  br label %bb11

bb11:                                             ; preds = %bb16, %bb10
  %i1.0 = phi i64 [ 0, %bb10 ], [ %tmp17, %bb16 ]
  %exitcond = icmp ne i64 %i1.0, 100
  br i1 %exitcond, label %bb12, label %bb18

bb12:                                             ; preds = %bb11
  %tmp13 = getelementptr inbounds float, float* %A, i64 %i1.0
  %tmp14 = load float, float* %tmp13, align 4
  %tmp15 = fadd float %tmp14, 1.000000e+01
  store float %tmp15, float* %tmp13, align 4
  br label %bb16

bb16:                                             ; preds = %bb12
  %tmp17 = add nuw nsw i64 %i1.0, 1
  br label %bb11

bb18:                                             ; preds = %bb11
  ret void
}
