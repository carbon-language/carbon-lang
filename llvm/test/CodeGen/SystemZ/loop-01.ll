; Test loop tuning.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Test that strength reduction is applied to addresses with a scale factor,
; but that indexed addressing can still be used.
define void @f1(i32 *%dest, i32 %a) {
; CHECK-LABEL: f1
; CHECK-NOT: sllg
; CHECK: st %r3, 0({{%r[1-5],%r[1-5]}})
; CHECK: br %r14
entry:
  br label %loop

loop:
  %index = phi i64 [ 0, %entry ], [ %next, %loop ]
  %ptr = getelementptr i32 *%dest, i64 %index
  store i32 %a, i32 *%ptr
  %next = add i64 %index, 1
  %cmp = icmp ne i64 %next, 100
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
