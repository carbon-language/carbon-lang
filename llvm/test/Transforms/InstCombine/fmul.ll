; RUN: opt -S -instcombine < %s | FileCheck %s

; (-0.0 - X) * C => X * -C
define float @test1(float %x) {
  %sub = fsub float -0.000000e+00, %x
  %mul = fmul float %sub, 2.0e+1
  ret float %mul

; CHECK-LABEL: @test1(
; CHECK: fmul float %x, -2.000000e+01
}

; (0.0 - X) * C => X * -C
define float @test2(float %x) {
  %sub = fsub nsz float 0.000000e+00, %x
  %mul = fmul float %sub, 2.0e+1
  ret float %mul

; CHECK-LABEL: @test2(
; CHECK: fmul float %x, -2.000000e+01
}

; (-0.0 - X) * (-0.0 - Y) => X * Y
define float @test3(float %x, float %y) {
  %sub1 = fsub float -0.000000e+00, %x
  %sub2 = fsub float -0.000000e+00, %y
  %mul = fmul fast float %sub1, %sub2
  ret float %mul
; CHECK-LABEL: @test3(
; CHECK: fmul fast float %x, %y
}

; (0.0 - X) * (0.0 - Y) => X * Y
define float @test4(float %x, float %y) {
  %sub1 = fsub nsz float 0.000000e+00, %x
  %sub2 = fsub nsz float 0.000000e+00, %y
  %mul = fmul float %sub1, %sub2
  ret float %mul
; CHECK-LABEL: @test4(
; CHECK: fmul float %x, %y
}

; (-0.0 - X) * Y => -0.0 - (X * Y)
define float @test5(float %x, float %y) {
  %sub1 = fsub float -0.000000e+00, %x
  %mul = fmul float %sub1, %y
  ret float %mul
; CHECK-LABEL: @test5(
; CHECK: %1 = fmul float %x, %y
; CHECK: %mul = fsub float -0.000000e+00, %1
}

; (0.0 - X) * Y => 0.0 - (X * Y)
define float @test6(float %x, float %y) {
  %sub1 = fsub nsz float 0.000000e+00, %x
  %mul = fmul float %sub1, %y
  ret float %mul
; CHECK-LABEL: @test6(
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
; CHECK-LABEL: @test7(
; CHECK: fsub float -0.000000e+00, %x
}

; Don't crash when attempting to cast a constant FMul to an instruction.
define void @test8(i32* %inout) {
entry:
  %0 = load i32* %inout, align 4
  %conv = uitofp i32 %0 to float
  %vecinit = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float undef>, float %conv, i32 3
  %sub = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vecinit
  %1 = shufflevector <4 x float> %sub, <4 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mul = fmul <4 x float> zeroinitializer, %1
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %local_var_7.0 = phi <4 x float> [ %mul, %entry ], [ %2, %for.body ]
  br i1 undef, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = insertelement <4 x float> %local_var_7.0, float 0.000000e+00, i32 2
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; X * -1.0 => -0.0 - X
define float @test9(float %x) {
  %mul = fmul float %x, -1.0
  ret float %mul

; CHECK-LABEL: @test9(
; CHECK-NOT: fmul
; CHECK: fsub
}

; PR18532
define <4 x float> @test10(<4 x float> %x) {
  %mul = fmul <4 x float> %x, <float -1.0, float -1.0, float -1.0, float -1.0>
  ret <4 x float> %mul

; CHECK-LABEL: @test10(
; CHECK-NOT: fmul
; CHECK: fsub
}
