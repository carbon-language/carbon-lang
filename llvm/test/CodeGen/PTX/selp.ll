; RUN: llc < %s -march=ptx32 | FileCheck %s

define ptx_device i32 @test_selp_i32(i1 %x, i32 %y, i32 %z) {
; CHECK: selp.u32 r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}, p{{[0-9]+}};
	%a = select i1 %x, i32 %y, i32 %z
	ret i32 %a
}

define ptx_device i64 @test_selp_i64(i1 %x, i64 %y, i64 %z) {
; CHECK: selp.u64 rd{{[0-9]+}}, rd{{[0-9]+}}, rd{{[0-9]+}}, p{{[0-9]+}};
	%a = select i1 %x, i64 %y, i64 %z
	ret i64 %a
}

define ptx_device float @test_selp_f32(i1 %x, float %y, float %z) {
; CHECK: selp.f32 r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}, p{{[0-9]+}};
	%a = select i1 %x, float %y, float %z
	ret float %a
}

define ptx_device double @test_selp_f64(i1 %x, double %y, double %z) {
; CHECK: selp.f64 rd{{[0-9]+}}, rd{{[0-9]+}}, rd{{[0-9]+}}, p{{[0-9]+}};
	%a = select i1 %x, double %y, double %z
	ret double %a
}
