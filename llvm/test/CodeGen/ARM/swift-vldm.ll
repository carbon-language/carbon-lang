; RUN: llc < %s -mcpu=swift -mtriple=armv7s-apple-ios | FileCheck %s

; vldm with registers not aligned with q registers need more micro-ops so that
; so that there usage becomes unbeneficial on swift.

; CHECK-LABEL: test_vldm
; CHECK: vldmia  r1, {d18, d19, d20}
; CHECK-NOT: vldmia  r1, {d17, d18, d19, d20}

define double @test_vldm(double %a, double %b, double* nocapture %x) {
entry:
  %mul73 = fmul double %a, %b
  %addr1 = getelementptr double * %x, i32 1
  %addr2 = getelementptr double * %x, i32 2
  %addr3 = getelementptr double * %x, i32 3
  %load0 = load double * %x
  %load1 = load double * %addr1
  %load2 = load double * %addr2
  %load3 = load double * %addr3
  %sub = fsub double %mul73, %load1
  %mul = fmul double %mul73, %load0
  %add = fadd double %mul73, %load2
  %div = fdiv double %mul73, %load3
  %red = fadd double %sub, %mul
  %red2 = fadd double %div, %add
  %red3 = fsub double %red, %red2
  ret double %red3
}
