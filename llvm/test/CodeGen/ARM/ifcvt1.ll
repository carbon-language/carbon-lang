; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s

define i32 @t1(i32 %a, i32 %b) {
; CHECK: t1:
	%tmp2 = icmp eq i32 %a, 0
	br i1 %tmp2, label %cond_false, label %cond_true

cond_true:
; CHECK: subeq r0, r1, #1
	%tmp5 = add i32 %b, 1
	ret i32 %tmp5

cond_false:
; CHECK: addne r0, r1, #1
	%tmp7 = add i32 %b, -1
	ret i32 %tmp7
}
