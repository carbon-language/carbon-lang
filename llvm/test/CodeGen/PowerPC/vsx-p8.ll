; RUN: llc -mcpu=pwr8 -mattr=+power8-vector < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Unaligned loads/stores on P8 and later should use VSX where possible.

define <2 x double> @test28u(<2 x double>* %a) {
  %v = load <2 x double>* %a, align 8
  ret <2 x double> %v

; CHECK-LABEL: @test28u
; CHECK: lxvd2x 34, 0, 3
; CHECK: blr
}

define void @test29u(<2 x double>* %a, <2 x double> %b) {
  store <2 x double> %b, <2 x double>* %a, align 8
  ret void

; CHECK-LABEL: @test29u
; CHECK: stxvd2x 34, 0, 3
; CHECK: blr
}

define <4 x float> @test32u(<4 x float>* %a) {
  %v = load <4 x float>* %a, align 8
  ret <4 x float> %v

; CHECK-LABEL: @test32u
; CHECK: lxvw4x 34, 0, 3
; CHECK: blr
}

define void @test33u(<4 x float>* %a, <4 x float> %b) {
  store <4 x float> %b, <4 x float>* %a, align 8
  ret void

; CHECK-LABEL: @test33u
; CHECK: stxvw4x 34, 0, 3
; CHECK: blr
}

