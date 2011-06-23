; RUN: llc < %s -march=ptx32 | FileCheck %s

define ptx_device i32 @t1(i32 %x, i32 %y) {
; CHECK: shr.u32 r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
	%z = lshr i32 %x, %y
; CHECK: ret;
	ret i32 %z
}

define ptx_device i32 @t2(i32 %x) {
; CHECK: shr.u32 r{{[0-9]+}}, r{{[0-9]+}}, 3
	%z = lshr i32 %x, 3
; CHECK: ret;
	ret i32 %z
}

define ptx_device i32 @t3(i32 %x) {
; CHECK: shr.u32 r{{[0-9]+}}, 3, r{{[0-9]+}}
	%z = lshr i32 3, %x
; CHECK: ret;
	ret i32 %z
}

define ptx_device i32 @t4(i32 %x, i32 %y) {
; CHECK: shr.s32 r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
	%z = ashr i32 %x, %y
; CHECK: ret;
	ret i32 %z
}

define ptx_device i32 @t5(i32 %x) {
; CHECK: shr.s32 r{{[0-9]+}}, r{{[0-9]+}}, 3
	%z = ashr i32 %x, 3
; CHECK: ret;
	ret i32 %z
}

define ptx_device i32 @t6(i32 %x) {
; CHECK: shr.s32 r{{[0-9]+}}, -3, r{{[0-9]+}}
	%z = ashr i32 -3, %x
; CHECK: ret;
	ret i32 %z
}
