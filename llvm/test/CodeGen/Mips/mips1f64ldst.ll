; RUN: llc  < %s -march=mipsel -mcpu=mips1 | FileCheck %s -check-prefix=CHECK-EL
; RUN: llc  < %s -march=mips -mcpu=mips1 | FileCheck %s -check-prefix=CHECK-EB

@g1 = common global double 0.000000e+00, align 8
@g2 = common global double 0.000000e+00, align 8

define double @foo0(double %d0) nounwind {
entry:
; CHECK-EL: lw  $[[R0:[0-9]+]], %got($CPI0_0)
; CHECK-EL: lwc1  $f[[R1:[0-9]+]], %lo($CPI0_0)($[[R0]])
; CHECK-EL: lwc1  $f{{[0-9]+}}, %lo($CPI0_0+4)($[[R0]])
; CHECK-EL: add.d $f[[R2:[0-9]+]], $f12, $f[[R1]]
; CHECK-EL: lw  $[[R3:[0-9]+]], %got(g1)
; CHECK-EL: swc1  $f[[R2]], 0($[[R3]])
; CHECK-EL: swc1  $f{{[0-9]+}}, 4($[[R3]])
; CHECK-EL: lw  $[[R4:[0-9]+]], %got(g2)
; CHECK-EL: lwc1  $f0, 0($[[R4]])
; CHECK-EL: lwc1  $f1, 4($[[R4]])

; CHECK-EB: lw  $[[R0:[0-9]+]], %got($CPI0_0)
; CHECK-EB: lwc1  $f{{[0-9]+}}, %lo($CPI0_0)($[[R0]])
; CHECK-EB: lwc1  $f[[R1:[0-9]+]], %lo($CPI0_0+4)($[[R0]])
; CHECK-EB: add.d $f[[R2:[0-9]+]], $f12, $f[[R1]]
; CHECK-EB: lw  $[[R3:[0-9]+]], %got(g1)
; CHECK-EB: swc1  $f{{[0-9]+}}, 0($[[R3]])
; CHECK-EB: swc1  $f[[R2]], 4($[[R3]])
; CHECK-EB: lw  $[[R4:[0-9]+]], %got(g2)
; CHECK-EB: lwc1  $f1, 0($[[R4]])
; CHECK-EB: lwc1  $f0, 4($[[R4]])

  %add = fadd double %d0, 2.000000e+00
  store double %add, double* @g1, align 8
  %tmp1 = load double* @g2, align 8
  ret double %tmp1
}

