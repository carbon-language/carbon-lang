; RUN: llc < %s -march=arm -mattr=+vfp3 | FileCheck %s

define arm_apcscc float @t1(float %x) nounwind readnone optsize {
entry:
; CHECK: t1:
; CHECK: fconsts s1, 16
  %0 = fadd float %x, 4.000000e+00
  ret float %0
}

define arm_apcscc double @t2(double %x) nounwind readnone optsize {
entry:
; CHECK: t2:
; CHECK: fconstd d1, 8
  %0 = fadd double %x, 3.000000e+00
  ret double %0
}

define arm_apcscc double @t3(double %x) nounwind readnone optsize {
entry:
; CHECK: t3:
; CHECK: fconstd d1, 170
  %0 = fmul double %x, -1.300000e+01
  ret double %0
}

define arm_apcscc float @t4(float %x) nounwind readnone optsize {
entry:
; CHECK: t4:
; CHECK: fconsts s1, 184
  %0 = fmul float %x, -2.400000e+01
  ret float %0
}
