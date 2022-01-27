; RUN: opt %loadPolly -analyze -polly-scops < %s | FileCheck %s

; The SCoP contains a loop with multiple exit blocks (BBs after leaving
; the loop). The current implementation of deriving their domain derives
; only a common domain for all of the exit blocks. We disabled loops with
; multiple exit blocks until this is fixed.
; XFAIL: *

; CHECK: { Stmt_bb3[i0] -> [0, 0] };
; CHECK: { Stmt_bb2[] -> [1, 0] };

; Verify that we generate the correct schedule. In older versions of Polly,
; we generated an incorrect schedule:
;
;   { Stmt_bb3[i0] -> [1, 0]; Stmt_bb2[] -> [0, 0] }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @hoge() {
bb:
  br label %bb3

bb1:                                              ; preds = %bb5
  br label %bb6

bb2:                                              ; preds = %bb3
  %tmp = phi i64 [ %tmp4, %bb3 ]
  br label %bb6

bb3:                                              ; preds = %bb5, %bb
  %tmp4 = phi i64 [ 0, %bb ], [ 0, %bb5 ]
  br i1 false, label %bb5, label %bb2

bb5:                                              ; preds = %bb3
  br i1 false, label %bb3, label %bb1

bb6:                                              ; preds = %bb2, %bb1
  %tmp2 = phi i64 [ %tmp, %bb2 ], [ undef, %bb1 ]
  ret void
}
