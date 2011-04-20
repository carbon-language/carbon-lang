; RUN: llc < %s -march=ptx32 | FileCheck %s

;define ptx_device i32 @t1(i32 %x, i32 %y) {
;	%z = mul i32 %x, %y
;	ret i32 %z
;}

;define ptx_device i32 @t2(i32 %x) {
;	%z = mul i32 %x, 1
;	ret i32 %z
;}

define ptx_device float @t1_f32(float %x, float %y) {
; CHECK: mul.f32 f0, f1, f2
; CHECK-NEXT: ret;
  %z = fmul float %x, %y
  ret float %z
}

define ptx_device double @t1_f64(double %x, double %y) {
; CHECK: mul.f64 fd0, fd1, fd2
; CHECK-NEXT: ret;
  %z = fmul double %x, %y
  ret double %z
}

define ptx_device float @t2_f32(float %x) {
; CHECK: mul.f32 f0, f1, 0F40A00000;
; CHECK-NEXT: ret;
  %z = fmul float %x, 5.0
  ret float %z
}

define ptx_device double @t2_f64(double %x) {
; CHECK: mul.f64 fd0, fd1, 0D4014000000000000;
; CHECK-NEXT: ret;
  %z = fmul double %x, 5.0
  ret double %z
}
