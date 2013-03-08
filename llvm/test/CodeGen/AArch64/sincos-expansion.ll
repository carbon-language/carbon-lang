; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs < %s | FileCheck %s

define float @test_sincos_f32(float %f) {
  %sin = call float @sinf(float %f) readnone
  %cos = call float @cosf(float %f) readnone
; CHECK: bl cosf
; CHECK: bl sinf
  %val = fadd float %sin, %cos
  ret float %val
}

define double @test_sincos_f64(double %f) {
  %sin = call double @sin(double %f) readnone
  %cos = call double @cos(double %f) readnone
  %val = fadd double %sin, %cos
; CHECK: bl cos
; CHECK: bl sin
  ret double %val
}

define fp128 @test_sincos_f128(fp128 %f) {
  %sin = call fp128 @sinl(fp128 %f) readnone
  %cos = call fp128 @cosl(fp128 %f) readnone
  %val = fadd fp128 %sin, %cos
; CHECK: bl cosl
; CHECK: bl sinl
  ret fp128 %val
}

declare float  @sinf(float) readonly
declare double @sin(double) readonly
declare fp128 @sinl(fp128) readonly
declare float @cosf(float) readonly
declare double @cos(double) readonly
declare fp128 @cosl(fp128) readonly