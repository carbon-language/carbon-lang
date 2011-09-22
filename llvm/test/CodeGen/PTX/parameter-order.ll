; RUN: llc < %s -march=ptx32 | FileCheck %s

; CHECK: .func (.reg .b32 %ret{{[0-9]+}}) test_parameter_order (.reg .b32 %param{{[0-9]+}}, .reg .b32 %param{{[0-9]+}}, .reg .b32 %param{{[0-9]+}}, .reg .b32 %param{{[0-9]+}})
define ptx_device i32 @test_parameter_order(float %a, i32 %b, i32 %c, float %d) {
; CHECK: sub.u32 %ret{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
	%result = sub i32 %b, %c
	ret i32 %result
}
