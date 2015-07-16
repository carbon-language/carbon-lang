; RUN: llc -O3 < %s -mtriple=arm64-apple-ios7.0 | FileCheck %s --check-prefix=CHECK-INEXACT
; RUN: llc -O3 < %s -mtriple=aarch64-linux-gnu | FileCheck %s --check-prefix=CHECK-FAST

; CHECK-INEXACT-LABEL: test1:
; CHECK-INEXACT-DAG: frintm
; CHECK-INEXACT-DAG: frintx

; CHECK-FAST-LABEL: test1:
; CHECK-FAST: frintm
; CHECK-FAST-NOT: frintx
define float @test1(float %a) #0 {
entry:
  %call = tail call float @floorf(float %a) nounwind readnone
  ret float %call
}

declare float @floorf(float) nounwind readnone

; CHECK-INEXACT-LABEL: test2:
; CHECK-INEXACT: frintm
; CHECK-INEXACT: frintx

; CHECK-FAST-LABEL: test2:
; CHECK-FAST: frintm
; CHECK-FAST-NOT: frintx
define double @test2(double %a) #0 {
entry:
  %call = tail call double @floor(double %a) nounwind readnone
  ret double %call
}

declare double @floor(double) nounwind readnone

; CHECK-INEXACT-LABEL: test3:
; CHECK-INEXACT: frinti

; CHECK-FAST-LABEL: test3:
; CHECK-FAST: frinti
define float @test3(float %a) #0 {
entry:
  %call = tail call float @nearbyintf(float %a) nounwind readnone
  ret float %call
}

declare float @nearbyintf(float) nounwind readnone

; CHECK-INEXACT-LABEL: test4:
; CHECK-INEXACT: frinti

; CHECK-FAST-LABEL: test4:
; CHECK-FAST: frinti
define double @test4(double %a) #0 {
entry:
  %call = tail call double @nearbyint(double %a) nounwind readnone
  ret double %call
}

declare double @nearbyint(double) nounwind readnone

; CHECK-INEXACT-LABEL: test5:
; CHECK-INEXACT: frintp
; CHECK-INEXACT: frintx

; CHECK-FAST-LABEL: test5:
; CHECK-FAST: frintp
; CHECK-FAST-NOT: frintx
define float @test5(float %a) #0 {
entry:
  %call = tail call float @ceilf(float %a) nounwind readnone
  ret float %call
}

declare float @ceilf(float) nounwind readnone

; CHECK-INEXACT-LABEL: test6:
; CHECK-INEXACT: frintp
; CHECK-INEXACT: frintx

; CHECK-FAST-LABEL: test6:
; CHECK-FAST: frintp
; CHECK-FAST-NOT: frintx
define double @test6(double %a) #0 {
entry:
  %call = tail call double @ceil(double %a) nounwind readnone
  ret double %call
}

declare double @ceil(double) nounwind readnone

; CHECK-INEXACT-LABEL: test7:
; CHECK-INEXACT: frintx

; CHECK-FAST-LABEL: test7:
; CHECK-FAST: frintx
define float @test7(float %a) #0 {
entry:
  %call = tail call float @rintf(float %a) nounwind readnone
  ret float %call
}

declare float @rintf(float) nounwind readnone

; CHECK-INEXACT-LABEL: test8:
; CHECK-INEXACT: frintx

; CHECK-FAST-LABEL: test8:
; CHECK-FAST: frintx
define double @test8(double %a) #0 {
entry:
  %call = tail call double @rint(double %a) nounwind readnone
  ret double %call
}

declare double @rint(double) nounwind readnone

; CHECK-INEXACT-LABEL: test9:
; CHECK-INEXACT: frintz
; CHECK-INEXACT: frintx

; CHECK-FAST-LABEL: test9:
; CHECK-FAST: frintz
; CHECK-FAST-NOT: frintx
define float @test9(float %a) #0 {
entry:
  %call = tail call float @truncf(float %a) nounwind readnone
  ret float %call
}

declare float @truncf(float) nounwind readnone

; CHECK-INEXACT-LABEL: test10:
; CHECK-INEXACT: frintz
; CHECK-INEXACT: frintx

; CHECK-FAST-LABEL: test10:
; CHECK-FAST: frintz
; CHECK-FAST-NOT: frintx
define double @test10(double %a) #0 {
entry:
  %call = tail call double @trunc(double %a) nounwind readnone
  ret double %call
}

declare double @trunc(double) nounwind readnone

; CHECK-INEXACT-LABEL: test11:
; CHECK-INEXACT: frinta
; CHECK-INEXACT: frintx

; CHECK-FAST-LABEL: test11:
; CHECK-FAST: frinta
; CHECK-FAST-NOT: frintx
define float @test11(float %a) #0 {
entry:
  %call = tail call float @roundf(float %a) nounwind readnone
  ret float %call
}

declare float @roundf(float %a) nounwind readnone

; CHECK-INEXACT-LABEL: test12:
; CHECK-INEXACT: frinta
; CHECK-INEXACT: frintx

; CHECK-FAST-LABEL: test12:
; CHECK-FAST: frinta
; CHECK-FAST-NOT: frintx
define double @test12(double %a) #0 {
entry:
  %call = tail call double @round(double %a) nounwind readnone
  ret double %call
}

declare double @round(double %a) nounwind readnone

; CHECK-INEXACT-LABEL: test13:
; CHECK-INEXACT-NOT: frintx
; CHECK-INEXACT: frintm

; CHECK-FAST-LABEL: test13:
; CHECK-FAST-NOT: frintx
; CHECK-FAST: frintm
define float @test13(float %a) #1 {
entry:
  %call = tail call float @floorf(float %a) nounwind readnone
  ret float %call
}

; CHECK-INEXACT-LABEL: test14:
; CHECK-INEXACT-NOT: frintx
; CHECK-INEXACT: frintm

; CHECK-FAST-LABEL: test14:
; CHECK-FAST-NOT: frintx
; CHECK-FAST: frintm
define double @test14(double %a) #1 {
entry:
  %call = tail call double @floor(double %a) nounwind readnone
  ret double %call
}

; CHECK-INEXACT-LABEL: test15:
; CHECK-INEXACT-NOT: frintx
; CHECK-INEXACT: frintp

; CHECK-FAST-LABEL: test15:
; CHECK-FAST-NOT: frintx
; CHECK-FAST: frintp
define float @test15(float %a) #1 {
entry:
  %call = tail call float @ceilf(float %a) nounwind readnone
  ret float %call
}

; CHECK-INEXACT-LABEL: test16:
; CHECK-INEXACT-NOT: frintx
; CHECK-INEXACT: frintp

; CHECK-FAST-LABEL: test16:
; CHECK-FAST-NOT: frintx
; CHECK-FAST: frintp
define double @test16(double %a) #1 {
entry:
  %call = tail call double @ceil(double %a) nounwind readnone
  ret double %call
}

; CHECK-INEXACT-LABEL: test17:
; CHECK-INEXACT-NOT: frintx
; CHECK-INEXACT: frintz

; CHECK-FAST-LABEL: test17:
; CHECK-FAST-NOT: frintx
; CHECK-FAST: frintz
define float @test17(float %a) #1 {
entry:
  %call = tail call float @truncf(float %a) nounwind readnone
  ret float %call
}

; CHECK-INEXACT-LABEL: test18:
; CHECK-INEXACT-NOT: frintx
; CHECK-INEXACT: frintz

; CHECK-FAST-LABEL: test18:
; CHECK-FAST-NOT: frintx
; CHECK-FAST: frintz
define double @test18(double %a) #1 {
entry:
  %call = tail call double @trunc(double %a) nounwind readnone
  ret double %call
}

; CHECK-INEXACT-LABEL: test19:
; CHECK-INEXACT-NOT: frintx
; CHECK-INEXACT: frinta

; CHECK-FAST-LABEL: test19:
; CHECK-FAST-NOT: frintx
; CHECK-FAST: frinta
define float @test19(float %a) #1 {
entry:
  %call = tail call float @roundf(float %a) nounwind readnone
  ret float %call
}

; CHECK-INEXACT-LABEL: test20:
; CHECK-INEXACT-NOT: frintx
; CHECK-INEXACT: frinta

; CHECK-FAST-LABEL: test20:
; CHECK-FAST-NOT: frintx
; CHECK-FAST: frinta
define double @test20(double %a) #1 {
entry:
  %call = tail call double @round(double %a) nounwind readnone
  ret double %call
}



attributes #0 = { nounwind }
attributes #1 = { nounwind "unsafe-fp-math"="true" }
