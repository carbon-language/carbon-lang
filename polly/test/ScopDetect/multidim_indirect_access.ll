; RUN: opt %loadPolly -polly-detect-unprofitable -polly-detect -analyze < %s | FileCheck %s
;
; The outer loop of this function will correctly not be recognized with the
; message:
;
;   Non affine access function: (sext i32 %tmp to i64)
;
; The access A[x] might mistakenly be treated as a multidimensional access with
; dimension size x. This test will check that we correctly invalidate the
; region and do not detect an outer SCoP.
;
; FIXME:
; We should detect the inner region but the PHI node in the exit blocks
; prohibits that.
;
;    void f(int *A, long N) {
;      int j = 0;
;      while (N > j) {
;        int x = A[0];
;        int i = 1;
;        do {
;          A[x] = 42;
;          A += x;
;        } while (i++ < N);
;      }
;    }
;
; CHECK-NOT: Valid Region for Scop: bb0 => bb13
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i64 %N) {
bb:
  br label %bb0

bb0:
  %j = phi i64 [ %j.next, %bb1 ], [ 1, %bb ]
  %tmp = load i32, i32* %A, align 4
  %exitcond0 = icmp sgt i64 %N, %j
  %j.next = add nuw nsw i64 %j, 1
  br i1 %exitcond0, label %bb1, label %bb13

bb1:                                              ; preds = %bb7, %bb0
  %i = phi i64 [ %i.next, %bb1 ], [ 1, %bb0 ]
  %.0 = phi i32* [ %A, %bb0 ], [ %tmp12, %bb1 ]
  %tmp8 = sext i32 %tmp to i64
  %tmp9 = getelementptr inbounds i32, i32* %.0, i64 %tmp8
  store i32 42, i32* %tmp9, align 4
  %tmp11 = sext i32 %tmp to i64
  %tmp12 = getelementptr inbounds i32, i32* %.0, i64 %tmp11
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp ne i64 %i, %N
  br i1 %exitcond, label %bb1, label %bb0

bb13:                                             ; preds = %bb1
  ret void
}
