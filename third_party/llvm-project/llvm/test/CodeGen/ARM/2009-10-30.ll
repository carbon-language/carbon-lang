; RUN: llc < %s  -mtriple=armv6-linux-gnueabi  | FileCheck %s
; This test checks that the address of the varg arguments is correctly
; computed when there are 5 or more regular arguments.

define void @f(i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, ...) {
entry:
;CHECK: sub	sp, sp, #4
;CHECK: add	r{{[0-9]+}}, sp, #8
;CHECK: str	r{{[0-9]+}}, [sp], #4
;CHECK: bx	lr
	%ap = alloca i8*, align 4
	%ap1 = bitcast i8** %ap to i8*
	call void @llvm.va_start(i8* %ap1)
	ret void
}

declare void @llvm.va_start(i8*) nounwind
