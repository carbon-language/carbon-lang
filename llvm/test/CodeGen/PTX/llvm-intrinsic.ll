; RUN: llc < %s -march=ptx32 -mattr=+ptx20 | FileCheck %s

define ptx_device float @test_sqrt_f32(float %x) {
entry:
; CHECK: sqrt.rn.f32 r{{[0-9]+}}, r{{[0-9]+}};
; CHECK-NEXT: ret;
  %y = call float @llvm.sqrt.f32(float %x)
  ret float %y
}

define ptx_device double @test_sqrt_f64(double %x) {
entry:
; CHECK: sqrt.rn.f64 rd{{[0-9]+}}, rd{{[0-9]+}};
; CHECK-NEXT: ret;
  %y = call double @llvm.sqrt.f64(double %x)
  ret double %y
}

define ptx_device float @test_sin_f32(float %x) {
entry:
; CHECK: sin.approx.f32 r{{[0-9]+}}, r{{[0-9]+}};
; CHECK-NEXT: ret;
  %y = call float @llvm.sin.f32(float %x)
  ret float %y
}

define ptx_device double @test_sin_f64(double %x) {
entry:
; CHECK: sin.approx.f64 rd{{[0-9]+}}, rd{{[0-9]+}};
; CHECK-NEXT: ret;
  %y = call double @llvm.sin.f64(double %x)
  ret double %y
}

define ptx_device float @test_cos_f32(float %x) {
entry:
; CHECK: cos.approx.f32 r{{[0-9]+}}, r{{[0-9]+}};
; CHECK-NEXT: ret;
  %y = call float @llvm.cos.f32(float %x)
  ret float %y
}

define ptx_device double @test_cos_f64(double %x) {
entry:
; CHECK: cos.approx.f64 rd{{[0-9]+}}, rd{{[0-9]+}};
; CHECK-NEXT: ret;
  %y = call double @llvm.cos.f64(double %x)
  ret double %y
}

declare float  @llvm.sqrt.f32(float)
declare double @llvm.sqrt.f64(double)
declare float  @llvm.sin.f32(float)
declare double @llvm.sin.f64(double)
declare float  @llvm.cos.f32(float)
declare double @llvm.cos.f64(double)
