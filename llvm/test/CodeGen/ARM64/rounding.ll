; RUN: llc -O3 < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64"
target triple = "arm64-apple-ios6.0.0"

; CHECK: test1
; CHECK: frintx
; CHECK: frintm
define float @test1(float %a) #0 {
entry:
  %call = tail call float @floorf(float %a) nounwind readnone
  ret float %call
}

declare float @floorf(float) nounwind readnone

; CHECK: test2
; CHECK: frintx
; CHECK: frintm
define double @test2(double %a) #0 {
entry:
  %call = tail call double @floor(double %a) nounwind readnone
  ret double %call
}

declare double @floor(double) nounwind readnone

; CHECK: test3
; CHECK: frinti
define float @test3(float %a) #0 {
entry:
  %call = tail call float @nearbyintf(float %a) nounwind readnone
  ret float %call
}

declare float @nearbyintf(float) nounwind readnone

; CHECK: test4
; CHECK: frinti
define double @test4(double %a) #0 {
entry:
  %call = tail call double @nearbyint(double %a) nounwind readnone
  ret double %call
}

declare double @nearbyint(double) nounwind readnone

; CHECK: test5
; CHECK: frintx
; CHECK: frintp
define float @test5(float %a) #0 {
entry:
  %call = tail call float @ceilf(float %a) nounwind readnone
  ret float %call
}

declare float @ceilf(float) nounwind readnone

; CHECK: test6
; CHECK: frintx
; CHECK: frintp
define double @test6(double %a) #0 {
entry:
  %call = tail call double @ceil(double %a) nounwind readnone
  ret double %call
}

declare double @ceil(double) nounwind readnone

; CHECK: test7
; CHECK: frintx
define float @test7(float %a) #0 {
entry:
  %call = tail call float @rintf(float %a) nounwind readnone
  ret float %call
}

declare float @rintf(float) nounwind readnone

; CHECK: test8
; CHECK: frintx
define double @test8(double %a) #0 {
entry:
  %call = tail call double @rint(double %a) nounwind readnone
  ret double %call
}

declare double @rint(double) nounwind readnone

; CHECK: test9
; CHECK: frintx
; CHECK: frintz
define float @test9(float %a) #0 {
entry:
  %call = tail call float @truncf(float %a) nounwind readnone
  ret float %call
}

declare float @truncf(float) nounwind readnone

; CHECK: test10
; CHECK: frintx
; CHECK: frintz
define double @test10(double %a) #0 {
entry:
  %call = tail call double @trunc(double %a) nounwind readnone
  ret double %call
}

declare double @trunc(double) nounwind readnone

; CHECK: test11
; CHECK: frintx
; CHECK: frinta
define float @test11(float %a) #0 {
entry:
  %call = tail call float @roundf(float %a) nounwind readnone
  ret float %call
}

declare float @roundf(float %a) nounwind readnone

; CHECK: test12
; CHECK: frintx
; CHECK: frinta
define double @test12(double %a) #0 {
entry:
  %call = tail call double @round(double %a) nounwind readnone
  ret double %call
}

declare double @round(double %a) nounwind readnone

; CHECK: test13
; CHECK-NOT: frintx
; CHECK: frintm
define float @test13(float %a) #1 {
entry:
  %call = tail call float @floorf(float %a) nounwind readnone
  ret float %call
}

; CHECK: test14
; CHECK-NOT: frintx
; CHECK: frintm
define double @test14(double %a) #1 {
entry:
  %call = tail call double @floor(double %a) nounwind readnone
  ret double %call
}

; CHECK: test15
; CHECK-NOT: frintx
; CHECK: frintp
define float @test15(float %a) #1 {
entry:
  %call = tail call float @ceilf(float %a) nounwind readnone
  ret float %call
}

; CHECK: test16
; CHECK-NOT: frintx
; CHECK: frintp
define double @test16(double %a) #1 {
entry:
  %call = tail call double @ceil(double %a) nounwind readnone
  ret double %call
}

; CHECK: test17
; CHECK-NOT: frintx
; CHECK: frintz
define float @test17(float %a) #1 {
entry:
  %call = tail call float @truncf(float %a) nounwind readnone
  ret float %call
}

; CHECK: test18
; CHECK-NOT: frintx
; CHECK: frintz
define double @test18(double %a) #1 {
entry:
  %call = tail call double @trunc(double %a) nounwind readnone
  ret double %call
}

; CHECK: test19
; CHECK-NOT: frintx
; CHECK: frinta
define float @test19(float %a) #1 {
entry:
  %call = tail call float @roundf(float %a) nounwind readnone
  ret float %call
}

; CHECK: test20
; CHECK-NOT: frintx
; CHECK: frinta
define double @test20(double %a) #1 {
entry:
  %call = tail call double @round(double %a) nounwind readnone
  ret double %call
}



attributes #0 = { nounwind }
attributes #1 = { nounwind "unsafe-fp-math"="true" }
