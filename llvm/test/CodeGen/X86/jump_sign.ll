; RUN: llc < %s -march=x86 | FileCheck %s

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
