; RUN: opt %loadPolly -polly-detect-unprofitable -polly-detect -analyze < %s | FileCheck %s --check-prefix=INDEPENDENT
; RUN: opt %loadPolly -polly-detect-unprofitable -polly-detect -analyze  < %s | FileCheck %s --check-prefix=NON_INDEPENDENT
;
; With the IndependentBlocks and PollyPrepare passes this will __correctly__
; not be recognized as a SCoP and the debug states:
;
;   SCEV of PHI node refers to SSA names in region
;
; Without IndependentBlocks and PollyPrepare the access A[x] is mistakenly
; treated as a multidimensional access with dimension size x. This test will
; check that we correctly invalidate the region and do not detect a outer SCoP.
;
; FIXME:
; We should detect the inner region but the PHI node in the exit blocks that.
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
; INDEPENDENT-NOT: Valid
; NON_INDEPENDENT-NOT: Valid
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
