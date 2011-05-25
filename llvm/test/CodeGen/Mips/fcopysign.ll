; RUN: llc  < %s -march=mipsel -mcpu=4ke | FileCheck %s -check-prefix=CHECK-EL
; RUN: llc  < %s -march=mips -mcpu=4ke | FileCheck %s -check-prefix=CHECK-EB

define double @func0(double %d0, double %d1) nounwind readnone {
entry:
; CHECK-EL: func0:
; CHECK-EL: lui $[[T0:[0-9]+]], 32767
; CHECK-EL: lui $[[T1:[0-9]+]], 32768
; CHECK-EL: mfc1 $[[HI0:[0-9]+]], $f13
; CHECK-EL: ori $[[MSK0:[0-9]+]], $[[T0]], 65535
; CHECK-EL: mfc1 $[[HI1:[0-9]+]], $f15
; CHECK-EL: ori $[[MSK1:[0-9]+]], $[[T1]], 0
; CHECK-EL: and $[[AND0:[0-9]+]], $[[HI0]], $[[MSK0]]
; CHECK-EL: and $[[AND1:[0-9]+]], $[[HI1]], $[[MSK1]]
; CHECK-EL: mfc1 $[[LO0:[0-9]+]], $f12
; CHECK-EL: or  $[[OR:[0-9]+]], $[[AND0]], $[[AND1]]
; CHECK-EL: mtc1 $[[LO0]], $f0
; CHECK-EL: mtc1 $[[OR]], $f1
;
; CHECK-EB: lui $[[T0:[0-9]+]], 32767
; CHECK-EB: lui $[[T1:[0-9]+]], 32768
; CHECK-EB: mfc1 $[[HI0:[0-9]+]], $f12
; CHECK-EB: ori $[[MSK0:[0-9]+]], $[[T0]], 65535
; CHECK-EB: mfc1 $[[HI1:[0-9]+]], $f14
; CHECK-EB: ori $[[MSK1:[0-9]+]], $[[T1]], 0
; CHECK-EB: and $[[AND0:[0-9]+]], $[[HI0]], $[[MSK0]]
; CHECK-EB: and $[[AND1:[0-9]+]], $[[HI1]], $[[MSK1]]
; CHECK-EB: or  $[[OR:[0-9]+]], $[[AND0]], $[[AND1]]
; CHECK-EB: mfc1 $[[LO0:[0-9]+]], $f13
; CHECK-EB: mtc1 $[[OR]], $f0
; CHECK-EB: mtc1 $[[LO0]], $f1
  %call = tail call double @copysign(double %d0, double %d1) nounwind readnone
  ret double %call
}

declare double @copysign(double, double) nounwind readnone

define float @func1(float %f0, float %f1) nounwind readnone {
entry:
; CHECK-EL: func1:
; CHECK-EL: lui $[[T0:[0-9]+]], 32767
; CHECK-EL: lui $[[T1:[0-9]+]], 32768
; CHECK-EL: mfc1 $[[ARG0:[0-9]+]], $f12
; CHECK-EL: ori $[[MSK0:[0-9]+]], $[[T0]], 65535
; CHECK-EL: mfc1 $[[ARG1:[0-9]+]], $f14
; CHECK-EL: ori $[[MSK1:[0-9]+]], $[[T1]], 0
; CHECK-EL: and $[[T2:[0-9]+]], $[[ARG0]], $[[MSK0]]
; CHECK-EL: and $[[T3:[0-9]+]], $[[ARG1]], $[[MSK1]]
; CHECK-EL: or  $[[T4:[0-9]+]], $[[T2]], $[[T3]]
; CHECK-EL: mtc1 $[[T4]], $f0
  %call = tail call float @copysignf(float %f0, float %f1) nounwind readnone
  ret float %call
}

declare float @copysignf(float, float) nounwind readnone
