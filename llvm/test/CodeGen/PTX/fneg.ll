; RUN: llc < %s -march=ptx32 | FileCheck %s

define ptx_device float @t1_f32(float %x) {
; CHECK: neg.f32 r{{[0-9]+}}, r{{[0-9]+}};
; CHECK-NEXT: ret;
	%y = fsub float -0.000000e+00, %x
	ret float %y
}

define ptx_device double @t1_f64(double %x) {
; CHECK: neg.f64 rd{{[0-9]+}}, rd{{[0-9]+}};
; CHECK-NEXT: ret;
	%y = fsub double -0.000000e+00, %x
	ret double %y
}
