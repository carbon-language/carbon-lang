; RUN: llc  < %s -march=mipsel | FileCheck %s -check-prefix=MIPS32-EL
; RUN: llc  < %s -march=mips | FileCheck %s -check-prefix=MIPS32-EB
; RUN: llc  < %s -march=mips64el -mcpu=mips64 -mattr=n64 | FileCheck %s -check-prefix=MIPS64

define double @func0(double %d0, double %d1) nounwind readnone {
entry:
; MIPS32-EL: func0:
; MIPS32-EL: lui $[[T1:[0-9]+]], 32768
; MIPS32-EL: ori $[[MSK1:[0-9]+]], $[[T1]], 0
; MIPS32-EL: mfc1 $[[HI0:[0-9]+]], $f15
; MIPS32-EL: and $[[AND1:[0-9]+]], $[[HI0]], $[[MSK1]]
; MIPS32-EL: lui $[[T0:[0-9]+]], 32767
; MIPS32-EL: ori $[[MSK0:[0-9]+]], $[[T0]], 65535
; MIPS32-EL: mfc1 $[[HI1:[0-9]+]], $f13
; MIPS32-EL: and $[[AND0:[0-9]+]], $[[HI1]], $[[MSK0]]
; MIPS32-EL: or  $[[OR:[0-9]+]], $[[AND0]], $[[AND1]]
; MIPS32-EL: mfc1 $[[LO0:[0-9]+]], $f12
; MIPS32-EL: mtc1 $[[LO0]], $f0
; MIPS32-EL: mtc1 $[[OR]], $f1
;
; MIPS32-EB: lui $[[T1:[0-9]+]], 32768
; MIPS32-EB: ori $[[MSK1:[0-9]+]], $[[T1]], 0
; MIPS32-EB: mfc1 $[[HI1:[0-9]+]], $f14
; MIPS32-EB: and $[[AND1:[0-9]+]], $[[HI1]], $[[MSK1]]
; MIPS32-EB: lui $[[T0:[0-9]+]], 32767
; MIPS32-EB: ori $[[MSK0:[0-9]+]], $[[T0]], 65535
; MIPS32-EB: mfc1 $[[HI0:[0-9]+]], $f12
; MIPS32-EB: and $[[AND0:[0-9]+]], $[[HI0]], $[[MSK0]]
; MIPS32-EB: or  $[[OR:[0-9]+]], $[[AND0]], $[[AND1]]
; MIPS32-EB: mfc1 $[[LO0:[0-9]+]], $f13
; MIPS32-EB: mtc1 $[[OR]], $f0
; MIPS32-EB: mtc1 $[[LO0]], $f1

; MIPS64: dmfc1 $[[R0:[0-9]+]], $f13
; MIPS64: and $[[R1:[0-9]+]], $[[R0]], ${{[0-9]+}}
; MIPS64: dmfc1 $[[R2:[0-9]+]], $f12
; MIPS64: and $[[R3:[0-9]+]], $[[R2]], ${{[0-9]+}}
; MIPS64: or  $[[R4:[0-9]+]], $[[R3]], $[[R1]]
; MIPS64: dmtc1 $[[R4]], $f0
  %call = tail call double @copysign(double %d0, double %d1) nounwind readnone
  ret double %call
}

declare double @copysign(double, double) nounwind readnone

define float @func1(float %f0, float %f1) nounwind readnone {
entry:
; MIPS32-EL: func1:
; MIPS32-EL: lui $[[T1:[0-9]+]], 32768
; MIPS32-EL: ori $[[MSK1:[0-9]+]], $[[T1]], 0
; MIPS32-EL: mfc1 $[[ARG1:[0-9]+]], $f14
; MIPS32-EL: and $[[T3:[0-9]+]], $[[ARG1]], $[[MSK1]]
; MIPS32-EL: lui $[[T0:[0-9]+]], 32767
; MIPS32-EL: ori $[[MSK0:[0-9]+]], $[[T0]], 65535
; MIPS32-EL: mfc1 $[[ARG0:[0-9]+]], $f12
; MIPS32-EL: and $[[T2:[0-9]+]], $[[ARG0]], $[[MSK0]]
; MIPS32-EL: or  $[[T4:[0-9]+]], $[[T2]], $[[T3]]
; MIPS32-EL: mtc1 $[[T4]], $f0
  %call = tail call float @copysignf(float %f0, float %f1) nounwind readnone
  ret float %call
}

declare float @copysignf(float, float) nounwind readnone
