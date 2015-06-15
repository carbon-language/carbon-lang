; RUN: opt < %s -instcombine -value-tracking-dom-conditions -S | FileCheck %s

define i32 @dom_cond(i32 %a, i32 %b) {
; CHECK-LABEL: @dom_cond(
entry:
  %v = add i32 %a, %b
  %cond = icmp ule i32 %v, 7
  br i1 %cond, label %then, label %exit

then:
  %v2 = add i32 %v, 8
; CHECK: or i32 %v, 8
  br label %exit

exit:
  %v3 = phi i32 [ %v, %entry ], [ %v2, %then ]
  ret i32 %v3
}
