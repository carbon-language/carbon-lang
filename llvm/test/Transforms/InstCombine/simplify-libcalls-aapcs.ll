; RUN: opt -S < %s -instcombine | FileCheck %s

; When simplify-libcall change an intrinsic call to a call to a library
; routine, it needs to set the proper calling convention for callers
; which use ARM target specific calling conventions.
; rdar://16261856

target triple = "thumbv7-apple-ios7"

; Function Attrs: nounwind ssp
define arm_aapcs_vfpcc double @t(double %x) #0 {
entry:
; CHECK-LABEL: @t
; CHECK: call arm_aapcs_vfpcc double @__exp10
  %0 = call double @llvm.pow.f64(double 1.000000e+01, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone
declare double @llvm.pow.f64(double, double) #1

attributes #0 = { nounwind ssp }
attributes #1 = { nounwind readnone }
