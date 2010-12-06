; RUN: llc < %s -march=ptx | FileCheck %s

define ptx_device i32 @t1(i32 %x, i32 %y) {
; CHECK: shl.b32 r0, r1, r2
	%z = shl i32 %x, %y
; CHECK: ret;
	ret i32 %z
}

define ptx_device i32 @t2(i32 %x) {
; CHECK: shl.b32 r0, r1, 3
	%z = shl i32 %x, 3
; CHECK: ret;
	ret i32 %z
}

define ptx_device i32 @t3(i32 %x) {
; CHECK: shl.b32 r0, 3, r1
	%z = shl i32 3, %x
; CHECK: ret;
	ret i32 %z
}
