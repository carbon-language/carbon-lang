; RUN: opt -loop-reduce -S < %s | FileCheck %s
;
; <rdar://10619599> "SelectionDAGBuilder shouldn't visit PHI nodes!" assert.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-f128:128:128-n8:16:32"
target triple = "i386-apple-darwin"

; LSR should convert the inner loop (bb7.us) IV (j.01.us) into float*.
; This involves a nested AddRec, the outer AddRec's loop invariant components
; cannot find a preheader, so they should be expanded in the loop header
; (bb7.lr.ph.us) below the existing phi i.12.us.
; CHECK: @nopreheader
; CHECK: bb7.lr.ph.us:
; CHECK: %lsr.iv = phi float*
; CHECK: bb7.us:
; CHECK: %lsr.iv2 = phi float*
define void @nopreheader(float* nocapture %a, i32 %n) nounwind {
entry:
  %0 = sdiv i32 %n, undef
  indirectbr i8* undef, [label %bb10.preheader]

bb10.preheader:                                   ; preds = %bb4
  indirectbr i8* undef, [label %bb8.preheader.lr.ph, label %return]

bb8.preheader.lr.ph:                              ; preds = %bb10.preheader
  indirectbr i8* null, [label %bb7.lr.ph.us, label %bb9]

bb7.lr.ph.us:                                     ; preds = %bb9.us, %bb8.preheader.lr.ph
  %i.12.us = phi i32 [ %2, %bb9.us ], [ 0, %bb8.preheader.lr.ph ]
  %tmp30 = mul i32 %0, %i.12.us
  indirectbr i8* undef, [label %bb7.us]

bb7.us:                                           ; preds = %bb7.lr.ph.us, %bb7.us
  %j.01.us = phi i32 [ 0, %bb7.lr.ph.us ], [ %1, %bb7.us ]
  %tmp31 = add i32 %tmp30, %j.01.us
  %scevgep9 = getelementptr float* %a, i32 %tmp31
  store float undef, float* %scevgep9, align 1
  %1 = add nsw i32 %j.01.us, 1
  indirectbr i8* undef, [label %bb9.us, label %bb7.us]

bb9.us:                                           ; preds = %bb7.us
  %2 = add nsw i32 %i.12.us, 1
  indirectbr i8* undef, [label %bb7.lr.ph.us, label %return]

bb9:                                              ; preds = %bb9, %bb8.preheader.lr.ph
  indirectbr i8* undef, [label %bb9, label %return]

return:                                           ; preds = %bb9, %bb9.us, %bb10.preheader
  ret void
}
