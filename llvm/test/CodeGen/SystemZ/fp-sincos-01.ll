; Test that combined sin/cos library call is emitted when appropriate

; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s --check-prefix=CHECK-NOOPT
; RUN: llc < %s -mtriple=s390x-linux-gnu -enable-unsafe-fp-math | FileCheck %s --check-prefix=CHECK-OPT

define float @f1(float %x) {
; CHECK-OPT-LABEL: f1:
; CHECK-OPT: brasl %r14, sincosf@PLT
; CHECK-OPT: le %f0, 164(%r15)
; CHECK-OPT: aeb %f0, 160(%r15)

; CHECK-NOOPT-LABEL: f1:
; CHECK-NOOPT: brasl %r14, sinf@PLT
; CHECK-NOOPT: brasl %r14, cosf@PLT
  %tmp1 = call float @sinf(float %x)
  %tmp2 = call float @cosf(float %x)
  %add = fadd float %tmp1, %tmp2
  ret float %add
}

define double @f2(double %x) {
; CHECK-OPT-LABEL: f2:
; CHECK-OPT: brasl %r14, sincos@PLT
; CHECK-OPT: ld %f0, 168(%r15)
; CHECK-OPT: adb %f0, 160(%r15)

; CHECK-NOOPT-LABEL: f2:
; CHECK-NOOPT: brasl %r14, sin@PLT
; CHECK-NOOPT: brasl %r14, cos@PLT
  %tmp1 = call double @sin(double %x)
  %tmp2 = call double @cos(double %x)
  %add = fadd double %tmp1, %tmp2
  ret double %add
}

define fp128 @f3(fp128 %x) {
; CHECK-OPT-LABEL: f3:
; CHECK-OPT: brasl %r14, sincosl@PLT
; CHECK-OPT: axbr

; CHECK-NOOPT-LABEL: f3:
; CHECK-NOOPT: brasl %r14, sinl@PLT
; CHECK-NOOPT: brasl %r14, cosl@PLT
  %tmp1 = call fp128 @sinl(fp128 %x)
  %tmp2 = call fp128 @cosl(fp128 %x)
  %add = fadd fp128 %tmp1, %tmp2
  ret fp128 %add
}

declare float @sinf(float) readonly
declare double @sin(double) readonly
declare fp128 @sinl(fp128) readonly
declare float @cosf(float) readonly
declare double @cos(double) readonly
declare fp128 @cosl(fp128) readonly

