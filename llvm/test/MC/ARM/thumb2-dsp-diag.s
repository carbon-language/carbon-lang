; RUN: not llvm-mc -triple=thumbv7m < %s 2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

sxtab r0, r0, r0
sxtah r0, r0, r0
sxtab16 r0, r0, r0
sxtb16 r0, r0
sxtb16 r0, r0, ror #8
; CHECK-ERRORS: error: instruction requires: arm-mode
; CHECK-ERRORS: error: instruction requires: arm-mode
; CHECK-ERRORS: error: instruction requires: arm-mode
; CHECK-ERRORS: error: instruction requires: arm-mode
; CHECK-ERRORS: error: invalid operand for instruction

uxtab r0, r0, r0
uxtah r0, r0, r0
uxtab16 r0, r0, r0
uxtb16 r0, r0
uxtb16 r0, r0, ror #8
; CHECK-ERRORS: error: instruction requires: arm-mode
; CHECK-ERRORS: error: instruction requires: arm-mode
; CHECK-ERRORS: error: instruction requires: arm-mode
; CHECK-ERRORS: error: instruction requires: arm-mode
; CHECK-ERRORS: error: invalid operand for instruction
