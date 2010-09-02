; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -relocation-model=pic | FileCheck %s
; rdar://7387640

; This now reduces to a single induction variable.

; TODO: It still gets a GPR shuffle at the end of the loop
; This is because something in instruction selection has decided
; that comparing the pre-incremented value with zero is better
; than comparing the post-incremented value with -4.

@G = external global i32                          ; <i32*> [#uses=2]
@array = external global i32*                     ; <i32**> [#uses=1]

define void @t() nounwind optsize {
; CHECK: t:
; CHECK: mov.w r2, #1000
entry:
  %.pre = load i32* @G, align 4                   ; <i32> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %entry
; CHECK: LBB0_1:
; CHECK: cmp r2, #0
; CHECK: sub{{(.w)?}} [[REGISTER:(r[0-9]+)|(lr)]], r2, #1
; CHECK: mov r2, [[REGISTER]]

  %0 = phi i32 [ %.pre, %entry ], [ %3, %bb ]     ; <i32> [#uses=1]
  %indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ] ; <i32> [#uses=2]
  %tmp5 = sub i32 1000, %indvar                   ; <i32> [#uses=1]
  %1 = load i32** @array, align 4                 ; <i32*> [#uses=1]
  %scevgep = getelementptr i32* %1, i32 %tmp5     ; <i32*> [#uses=1]
  %2 = load i32* %scevgep, align 4                ; <i32> [#uses=1]
  %3 = add nsw i32 %2, %0                         ; <i32> [#uses=2]
  store i32 %3, i32* @G, align 4
  %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %indvar.next, 1001      ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb
  ret void
}
