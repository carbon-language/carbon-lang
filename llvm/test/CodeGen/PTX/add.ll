; RUN: llc < %s -march=ptx | FileCheck %s

define ptx_device i32 @t1(i32 %x, i32 %y) {
; CHECK: add.s32 r0, r1, r2;
	%z = add i32 %x, %y
; CHECK: ret;
	ret i32 %z
}

define ptx_device i32 @t2(i32 %x) {
; CHECK: add.s32 r0, r1, 1;
	%z = add i32 %x, 1
; CHECK: ret;
	ret i32 %z
}
