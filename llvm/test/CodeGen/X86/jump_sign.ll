; RUN: llc < %s -march=x86 -mcpu=pentiumpro | FileCheck %s

define i32 @f(i32 %X) {
entry:
; CHECK: f:
; CHECK: jns
	%tmp1 = add i32 %X, 1		; <i32> [#uses=1]
	%tmp = icmp slt i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	%tmp2 = tail call i32 (...)* @bar( )		; <i32> [#uses=0]
	br label %cond_next

cond_next:		; preds = %cond_true, %entry
	%tmp3 = tail call i32 (...)* @baz( )		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @bar(...)

declare i32 @baz(...)

; rdar://10633221
; rdar://11355268
define i32 @g(i32 %a, i32 %b) nounwind {
entry:
; CHECK: g:
; CHECK-NOT: test
; CHECK: cmovs
  %sub = sub nsw i32 %a, %b
  %cmp = icmp sgt i32 %sub, 0
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}

; rdar://10734411
define i32 @h(i32 %a, i32 %b) nounwind {
entry:
; CHECK: h:
; CHECK-NOT: cmp
; CHECK: cmov
; CHECK-NOT: movl
; CHECK: ret
  %cmp = icmp slt i32 %b, %a
  %sub = sub nsw i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}
define i32 @i(i32 %a, i32 %b) nounwind {
entry:
; CHECK: i:
; CHECK-NOT: cmp
; CHECK: cmov
; CHECK-NOT: movl
; CHECK: ret
  %cmp = icmp sgt i32 %a, %b
  %sub = sub nsw i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}
define i32 @j(i32 %a, i32 %b) nounwind {
entry:
; CHECK: j:
; CHECK-NOT: cmp
; CHECK: cmov
; CHECK-NOT: movl
; CHECK: ret
  %cmp = icmp ugt i32 %a, %b
  %sub = sub i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}
define i32 @k(i32 %a, i32 %b) nounwind {
entry:
; CHECK: k:
; CHECK-NOT: cmp
; CHECK: cmov
; CHECK-NOT: movl
; CHECK: ret
  %cmp = icmp ult i32 %b, %a
  %sub = sub i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}
; rdar://11540023
define i64 @n(i64 %x, i64 %y) nounwind {
entry:
; CHECK: n:
; CHECK-NOT: sub
; CHECK: cmp
  %sub = sub nsw i64 %x, %y
  %cmp = icmp slt i64 %sub, 0
  %y.x = select i1 %cmp, i64 %y, i64 %x
  ret i64 %y.x
}
