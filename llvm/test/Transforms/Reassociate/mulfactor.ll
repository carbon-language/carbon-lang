; RUN: opt < %s -reassociate -instcombine -S | FileCheck %s

define i32 @f(i32 %a, i32 %b) {
; CHECK: @f
; CHECK: mul
; CHECK: mul
; CHECK-NOT: mul
; CHECK: ret

entry:
	%tmp.2 = mul i32 %a, %a
	%tmp.5 = shl i32 %a, 1
	%tmp.6 = mul i32 %tmp.5, %b
	%tmp.10 = mul i32 %b, %b
	%tmp.7 = add i32 %tmp.6, %tmp.2
	%tmp.11 = add i32 %tmp.7, %tmp.10
	ret i32 %tmp.11
}

