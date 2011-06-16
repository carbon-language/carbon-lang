; RUN: llc < %s -march=ptx32 | FileCheck %s

; CHECK: .func (.reg .b32 r0) test_parameter_order (.reg .b32 r1, .reg .b32 r2, .reg .b32 r3, .reg .b32 r4)
define ptx_device i32 @test_parameter_order(float %a, i32 %b, i32 %c, float %d) {
; CHECK: sub.u32 r0, r2, r3
	%result = sub i32 %b, %c
	ret i32 %result
}
