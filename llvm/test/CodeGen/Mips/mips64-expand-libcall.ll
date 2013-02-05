; RUN: llc -march=mips64el -mcpu=mips64r2 -O3 < %s | FileCheck %s

; Check that %add is not passed in an integer register.
;
; CHECK-NOT: dmfc1 $4

define double @callfloor(double %d) nounwind readnone {
entry:
  %add = fadd double %d, 1.000000e+00
  %call = tail call double @floor(double %add) nounwind readnone
  ret double %call
}

declare double @floor(double) nounwind readnone
