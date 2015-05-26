; RUN: opt %loadPolly -polly-detect-unprofitable -polly-scops -polly-model-phi-nodes -disable-polly-intra-scop-scalar-to-array -analyze < %s | FileCheck %s
;
;    float f(float *A, int N) {
;      float tmp = 0;
;      for (int i = 0; i < N; i++)
;        tmp += A[i];
;    }
;
; CHECK:      Statements {
; CHECK-LABEL:   Stmt_bb1
; CHECK-NOT: Access
; CHECK:              ReadAccess := [Reduction Type: NONE]
; CHECK:                  [N] -> { Stmt_bb1[i0] -> MemRef_tmp_0[] };
; CHECK-NOT: Access
; CHECK:              MustWriteAccess :=  [Reduction Type: NONE]
; CHECK:                  [N] -> { Stmt_bb1[i0] -> MemRef_tmp_0[] };
; CHECK-NOT: Access
; CHECK-LABEL:   Stmt_bb4
; CHECK-NOT: Access
; CHECK:              MustWriteAccess :=  [Reduction Type: NONE]
; CHECK:                  [N] -> { Stmt_bb4[i0] -> MemRef_tmp_0[] };
; CHECK-NOT: Access
; CHECK:              ReadAccess := [Reduction Type: NONE]
; CHECK:                  [N] -> { Stmt_bb4[i0] -> MemRef_tmp_0[] };
; CHECK-NOT: Access
; CHECK:              ReadAccess := [Reduction Type: NONE]
; CHECK:                  [N] -> { Stmt_bb4[i0] -> MemRef_A[i0] };
; CHECK-NOT: Access
; CHECK:      }
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(float* %A, i32 %N) {
bb:
  %tmp = sext i32 %N to i64
  br label %bb1

bb1:                                              ; preds = %bb4, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb4 ], [ 0, %bb ]
  %tmp.0 = phi float [ 0.000000e+00, %bb ], [ %tmp7, %bb4 ]
  %tmp2 = icmp slt i64 %indvars.iv, %tmp
  br i1 %tmp2, label %bb3, label %bb8

bb3:                                              ; preds = %bb1
  br label %bb4

bb4:                                              ; preds = %bb3
  %tmp5 = getelementptr inbounds float, float* %A, i64 %indvars.iv
  %tmp6 = load float, float* %tmp5, align 4
  %tmp7 = fadd float %tmp.0, %tmp6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  br label %exit

exit:
  ret void
}
