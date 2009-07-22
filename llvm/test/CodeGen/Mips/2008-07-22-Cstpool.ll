; RUN: llvm-as < %s | llc -march=mips -f | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"

; CHECK: .section .rodata.cst4,"aMs",@progbits,4
; CHECK: $CPI1_0:
; CHECK: $CPI1_1:
; CHECK: F:
define float @F(float %a) nounwind {
entry:
	fadd float %a, 0x4011333340000000		; <float>:0 [#uses=1]
	fadd float %0, 0x4010666660000000		; <float>:1 [#uses=1]
	ret float %1
}
