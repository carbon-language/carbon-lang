; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s

; CHECK-LABEL test_fabsf(
define float @test_fabsf(float %f) {
; CHECK: abs.f32
  %x = call float @llvm.fabs.f32(float %f)
  ret float %x
}

; CHECK-LABEL: test_fabs(
define double @test_fabs(double %d) {
; CHECK: abs.f64
  %x = call double @llvm.fabs.f64(double %d)
  ret double %x
}

; CHECK-LABEL: test_nvvm_sqrt(
define float @test_nvvm_sqrt(float %a) {
; CHECK: sqrt.rn.f32
  %val = call float @llvm.nvvm.sqrt.f(float %a)
  ret float %val
}

; CHECK-LABEL: test_bitreverse32(
define i32 @test_bitreverse32(i32 %a) {
; CHECK: brev.b32
  %val = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %val
}

; CHECK-LABEL: test_bitreverse64(
define i64 @test_bitreverse64(i64 %a) {
; CHECK: brev.b64
  %val = call i64 @llvm.bitreverse.i64(i64 %a)
  ret i64 %val
}

declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare float @llvm.nvvm.sqrt.f(float)
declare i32 @llvm.bitreverse.i32(i32)
declare i64 @llvm.bitreverse.i64(i64)
