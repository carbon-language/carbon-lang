; RUN: opt < %s -constprop -S | FileCheck %s

define i32 @test1() {
  %A = bitcast i32 2139171423 to float
  %B = insertelement <1 x float> undef, float %A, i32 0
  %C = extractelement <1 x float> %B, i32 0
  %D = bitcast float %C to i32
  ret i32 %D
; CHECK: @test1
; CHECK: ret i32 2139171423
}

