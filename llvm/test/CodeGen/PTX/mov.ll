; RUN: llc < %s -march=ptx | FileCheck %s

define ptx_device i32 @t1() {
;CHECK: mov.s32 r0, 0;
;CHECK: ret;
	ret i32 0
}

define ptx_device i32 @t2(i32 %x) {
;CHECK: mov.s32 r0, r1;
;CHECK: ret;
	ret i32 %x
}
