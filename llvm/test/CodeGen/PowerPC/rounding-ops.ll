; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -enable-unsafe-fp-math | FileCheck -check-prefix=CHECK-FM %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define float @test1(float %x) nounwind  {
  %call = tail call float @floorf(float %x) nounwind readnone
  ret float %call

; CHECK: test1:
; CHECK: frim 1, 1

; CHECK-FM: test1:
; CHECK-FM: frim 1, 1
}

declare float @floorf(float) nounwind readnone

define double @test2(double %x) nounwind  {
  %call = tail call double @floor(double %x) nounwind readnone
  ret double %call

; CHECK: test2:
; CHECK: frim 1, 1

; CHECK-FM: test2:
; CHECK-FM: frim 1, 1
}

declare double @floor(double) nounwind readnone

define float @test3(float %x) nounwind  {
  %call = tail call float @nearbyintf(float %x) nounwind readnone
  ret float %call

; CHECK: test3:
; CHECK-NOT: frin

; CHECK-FM: test3:
; CHECK-FM: frin 1, 1
}

declare float @nearbyintf(float) nounwind readnone

define double @test4(double %x) nounwind  {
  %call = tail call double @nearbyint(double %x) nounwind readnone
  ret double %call

; CHECK: test4:
; CHECK-NOT: frin

; CHECK-FM: test4:
; CHECK-FM: frin 1, 1
}

declare double @nearbyint(double) nounwind readnone

define float @test5(float %x) nounwind  {
  %call = tail call float @ceilf(float %x) nounwind readnone
  ret float %call

; CHECK: test5:
; CHECK: frip 1, 1

; CHECK-FM: test5:
; CHECK-FM: frip 1, 1
}

declare float @ceilf(float) nounwind readnone

define double @test6(double %x) nounwind  {
  %call = tail call double @ceil(double %x) nounwind readnone
  ret double %call

; CHECK: test6:
; CHECK: frip 1, 1

; CHECK-FM: test6:
; CHECK-FM: frip 1, 1
}

declare double @ceil(double) nounwind readnone

define float @test9(float %x) nounwind  {
  %call = tail call float @truncf(float %x) nounwind readnone
  ret float %call

; CHECK: test9:
; CHECK: friz 1, 1

; CHECK-FM: test9:
; CHECK-FM: friz 1, 1
}

declare float @truncf(float) nounwind readnone

define double @test10(double %x) nounwind  {
  %call = tail call double @trunc(double %x) nounwind readnone
  ret double %call

; CHECK: test10:
; CHECK: friz 1, 1

; CHECK-FM: test10:
; CHECK-FM: friz 1, 1
}

declare double @trunc(double) nounwind readnone

define void @test11(float %x, float* %y) nounwind  {
  %call = tail call float @rintf(float %x) nounwind readnone
  store float %call, float* %y
  ret void

; CHECK: test11:
; CHECK-NOT: frin

; CHECK-FM: test11:
; CHECK-FM: frin [[R2:[0-9]+]], [[R1:[0-9]+]]
; CHECK-FM: fcmpu [[CR:[0-9]+]], [[R2]], [[R1]]
; CHECK-FM: beq [[CR]], .LBB[[BB:[0-9]+]]_2
; CHECK-FM: mtfsb1 6
; CHECK-FM: .LBB[[BB]]_2:
; CHECK-FM: blr
}

declare float @rintf(float) nounwind readnone

define void @test12(double %x, double* %y) nounwind  {
  %call = tail call double @rint(double %x) nounwind readnone
  store double %call, double* %y
  ret void

; CHECK: test12:
; CHECK-NOT: frin

; CHECK-FM: test12:
; CHECK-FM: frin [[R2:[0-9]+]], [[R1:[0-9]+]]
; CHECK-FM: fcmpu [[CR:[0-9]+]], [[R2]], [[R1]]
; CHECK-FM: beq [[CR]], .LBB[[BB:[0-9]+]]_2
; CHECK-FM: mtfsb1 6
; CHECK-FM: .LBB[[BB]]_2:
; CHECK-FM: blr
}

declare double @rint(double) nounwind readnone

