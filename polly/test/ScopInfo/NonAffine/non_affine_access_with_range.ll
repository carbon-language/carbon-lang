; RUN: opt %loadPolly -polly-detect-unprofitable -polly-scops -polly-allow-nonaffine -analyze < %s | FileCheck %s
;
;    void f(int *A, char c) {
;      for (int i = 0; i < 1024; i++)
;        A[i * c]++;
;    }
;
; CHECK: ReadAccess := [Reduction Type: +] [Scalar: 0]
; CHECK:     { Stmt_bb2[i0] -> MemRef_A[o0] : o0 <= 261115 and o0 >= -3 };
; CHECK: MayWriteAccess := [Reduction Type: +] [Scalar: 0]
; CHECK:     { Stmt_bb2[i0] -> MemRef_A[o0] : o0 <= 261115 and o0 >= -3 };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i8 %c) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb8, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb8 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb9

bb2:                                              ; preds = %bb1
  %tmp = zext i8 %c to i32
  %tmp3 = zext i32 %tmp to i64
  %tmp4 = mul nuw nsw i64 %indvars.iv, %tmp3
  %tmp4b = add nsw nuw i64 %tmp4, -3
  %tmp5 = getelementptr inbounds i32, i32* %A, i64 %tmp4b
  %tmp6 = load i32* %tmp5, align 4
  %tmp7 = add nsw i32 %tmp6, 1
  store i32 %tmp7, i32* %tmp5, align 4
  br label %bb8

bb8:                                              ; preds = %bb2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb9:                                              ; preds = %bb1
  ret void
}
