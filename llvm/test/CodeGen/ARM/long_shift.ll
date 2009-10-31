; RUN: llc < %s -march=arm | FileCheck %s

define i64 @f0(i64 %A, i64 %B) {
; CHECK: f0
; CHECK: rrx
	%tmp = bitcast i64 %A to i64
	%tmp2 = lshr i64 %B, 1
	%tmp3 = sub i64 %tmp, %tmp2
	ret i64 %tmp3
}

define i32 @f1(i64 %x, i64 %y) {
; CHECK: f1
; CHECK: mov r0, r0, lsl r2
	%a = shl i64 %x, %y
	%b = trunc i64 %a to i32
	ret i32 %b
}

define i32 @f2(i64 %x, i64 %y) {
; CHECK: f2
; CHECK: __ashrdi3
	%a = ashr i64 %x, %y
	%b = trunc i64 %a to i32
	ret i32 %b
}

define i32 @f3(i64 %x, i64 %y) {
; CHECK: f3
; CHECK: __lshrdi3
	%a = lshr i64 %x, %y
	%b = trunc i64 %a to i32
	ret i32 %b
}
