; RUN: opt -S -instcombine < %s | FileCheck %s

define float @test1(float %x) nounwind readnone ssp {
  %div = fdiv float %x, 0x3810000000000000
  ret float %div

; CHECK-LABEL: @test1(
; CHECK-NEXT: fmul float %x, 0x47D0000000000000
}

define float @test2(float %x) nounwind readnone ssp {
  %div = fdiv float %x, 0x47E0000000000000
  ret float %div

; CHECK-LABEL: @test2(
; CHECK-NEXT: fdiv float %x, 0x47E0000000000000
}

define float @test3(float %x) nounwind readnone ssp {
  %div = fdiv float %x, 0x36A0000000000000
  ret float %div

; CHECK-LABEL: @test3(
; CHECK-NEXT: fdiv float %x, 0x36A0000000000000
}

define float @test4(float %x) nounwind readnone ssp {
  %div = fdiv fast float %x, 8.0
  ret float %div

; CHECK-LABEL: @test4(
; CHECK-NEXT: fmul fast float %x, 1.250000e-01
}
