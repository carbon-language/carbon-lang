; RUN: opt %loadPolly %defaultOpts -print-scev-affine  -analyze  < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f(i32 %a, i32 %b, i32 %c, i32 %d, [4 x i32]* nocapture %x) nounwind {
entry:
  br label %bb2.preheader

bb1:                                              ; preds = %bb2.preheader, %bb1
  %indvar = phi i64 [ 0, %bb2.preheader ], [ %indvar.next, %bb1 ] ; <i64> [#uses=3]
; CHECK: 1 * {0,+,1}<%bb1> + 0 * 1
  %scevgep = getelementptr [4 x i32]* %x, i64 %indvar, i64 %0 ; <i32*> [#uses=1]
; CHECK: 16 * {0,+,1}<%bb1> + 4 * {0,+,1}<%bb2.preheader> + 1 * %x + 0 * 1
  %tmp = mul i64 %indvar, %0                      ; <i64> [#uses=1]
; CHECK: 1 * {0,+,{0,+,1}<%bb2.preheader>}<%bb1> + 0 * 1
  %tmp13 = trunc i64 %tmp to i32                  ; <i32> [#uses=1]
; CHECK: 1 * {0,+,{0,+,1}<%bb2.preheader>}<%bb1> + 0 * 1
  store i32 %tmp13, i32* %scevgep, align 4
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
; CHECK: 1 * {0,+,1}<%bb1> + 1 * 1
  %exitcond = icmp eq i64 %indvar.next, 64        ; <i1> [#uses=1]
  br i1 %exitcond, label %bb3, label %bb1

bb3:                                              ; preds = %bb1
  %indvar.next12 = add i64 %0, 1                  ; <i64> [#uses=2]
; CHECK: 1 * {0,+,1}<%bb2.preheader> + 1 * 1
  %exitcond14 = icmp eq i64 %indvar.next12, 64    ; <i1> [#uses=1]
  br i1 %exitcond14, label %bb5, label %bb2.preheader

bb2.preheader:                                    ; preds = %bb3, %entry
  %0 = phi i64 [ 0, %entry ], [ %indvar.next12, %bb3 ] ; <i64> [#uses=3]
; CHECK: 1 * {0,+,1}<%bb2.preheader> + 0 * 1
  br label %bb1

bb5:                                              ; preds = %bb3
  ret i32 %a
}
