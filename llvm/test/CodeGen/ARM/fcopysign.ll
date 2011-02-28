; RUN: llc < %s -mtriple=armv7-apple-darwin -mcpu=cortex-a8 | FileCheck %s -check-prefix=SOFT
; RUN: llc < %s -mtriple=armv7-gnueabi -float-abi=hard -mcpu=cortex-a8 | FileCheck %s -check-prefix=HARD

; rdar://8984306
define float @test1(float %x, float %y) nounwind {
entry:
; SOFT: test1:
; SOFT: lsr r1, r1, #31
; SOFT: bfi r0, r1, #31, #1

; HARD: test1:
; HARD: vmov.i32 [[REG1:(d[0-9]+)]], #0x80000000
; HARD: vbsl [[REG1]], d2, d0
  %0 = tail call float @copysignf(float %x, float %y) nounwind
  ret float %0
}

define double @test2(double %x, double %y) nounwind {
entry:
; SOFT: test2:
; SOFT: lsr r2, r3, #31
; SOFT: bfi r1, r2, #31, #1

; HARD: test2:
; HARD: vmov.i32 [[REG2:(d[0-9]+)]], #0x80000000
; HARD: vshl.i64 [[REG2]], [[REG2]], #32
; HARD: vbsl [[REG2]], d1, d0
  %0 = tail call double @copysign(double %x, double %y) nounwind
  ret double %0
}

define double @test3(double %x, double %y, double %z) nounwind {
entry:
; SOFT: test3:
; SOFT: vmov.i32 [[REG3:(d[0-9]+)]], #0x80000000
; SOFT: vshl.i64 [[REG3]], [[REG3]], #32
; SOFT: vbsl [[REG3]],
  %0 = fmul double %x, %y
  %1 = tail call double @copysign(double %0, double %z) nounwind
  ret double %1
}

; rdar://9059537
define i32 @test4() ssp {
entry:
; SOFT: test4:
; SOFT: vcvt.f32.f64 s0, 
; SOFT: vmov.i32 [[REG4:(d[0-9]+)]], #0x80000000
; SOFT: vbic [[REG5:(d[0-9]+))], d0, [[REG4]]
; SOFT: vorr d0, [[REG4]], [[REG5]]
  %call80 = tail call double @copysign(double 1.000000e+00, double undef)
  %conv81 = fptrunc double %call80 to float
  %tmp88 = bitcast float %conv81 to i32
  ret i32 %tmp88
}

declare double @copysign(double, double) nounwind
declare float @copysignf(float, float) nounwind
