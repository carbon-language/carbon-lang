; RUN: llc < %s -march=arm | FileCheck %s

define i32 @f(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: _f:
; CHECK: subs
; CHECK-NOT: cmp
  %cmp = icmp sgt i32 %a, %b
  %sub = sub nsw i32 %a, %b
  %sub. = select i1 %cmp, i32 %sub, i32 0
  ret i32 %sub.
}

define i32 @g(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: _g:
; CHECK: subs
; CHECK-NOT: cmp
  %cmp = icmp slt i32 %a, %b
  %sub = sub nsw i32 %b, %a
  %sub. = select i1 %cmp, i32 %sub, i32 0
  ret i32 %sub.
}

define i32 @h(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: _h:
; CHECK: subs
; CHECK-NOT: cmp
  %cmp = icmp sgt i32 %a, 3
  %sub = sub nsw i32 %a, 3
  %sub. = select i1 %cmp, i32 %sub, i32 %b
  ret i32 %sub.
}
