; RUN: opt -S -instcombine < %s | FileCheck %s

; (-0.0 - X) * C => X * -C
define float @test1(float %x) {
  %sub = fsub float -0.000000e+00, %x
  %mul = fmul float %sub, 2.0e+1
  ret float %mul

; CHECK: @test1
; CHECK: fmul float %x, -2.000000e+01
}

; (0.0 - X) * C => X * -C
define float @test2(float %x) {
  %sub = fsub nsz float 0.000000e+00, %x
  %mul = fmul float %sub, 2.0e+1
  ret float %mul

; CHECK: @test2
; CHECK: fmul float %x, -2.000000e+01
}

; (-0.0 - X) * (-0.0 - Y) => X * Y
define float @test3(float %x, float %y) {
  %sub1 = fsub float -0.000000e+00, %x
  %sub2 = fsub float -0.000000e+00, %y
  %mul = fmul float %sub1, %sub2
  ret float %mul
; CHECK: @test3
; CHECK: fmul float %x, %y
}

; (0.0 - X) * (0.0 - Y) => X * Y
define float @test4(float %x, float %y) {
  %sub1 = fsub nsz float 0.000000e+00, %x
  %sub2 = fsub nsz float 0.000000e+00, %y
  %mul = fmul float %sub1, %sub2
  ret float %mul
; CHECK: @test4
; CHECK: fmul float %x, %y
}

; (-0.0 - X) * Y => -0.0 - (X * Y)
define float @test5(float %x, float %y) {
  %sub1 = fsub float -0.000000e+00, %x
  %mul = fmul float %sub1, %y
  ret float %mul
; CHECK: @test5
; CHECK: %1 = fmul float %x, %y
; CHECK: %mul = fsub float -0.000000e+00, %1
}

; (0.0 - X) * Y => 0.0 - (X * Y)
define float @test6(float %x, float %y) {
  %sub1 = fsub nsz float 0.000000e+00, %x
  %mul = fmul float %sub1, %y
  ret float %mul
; CHECK: @test6
; CHECK: %1 = fmul float %x, %y
; CHECK: %mul = fsub float -0.000000e+00, %1
}

; "(-0.0 - X) * Y => -0.0 - (X * Y)" is disabled if expression "-0.0 - X"
; has multiple uses.
define float @test7(float %x, float %y) {
  %sub1 = fsub float -0.000000e+00, %x
  %mul = fmul float %sub1, %y
  %mul2 = fmul float %mul, %sub1
  ret float %mul2
; CHECK: @test7
; CHECK: fsub float -0.000000e+00, %x
}
