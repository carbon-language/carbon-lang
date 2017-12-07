# RUN: not llvm-mc -triple=riscv32 -mattr=+c < %s 2>&1 | FileCheck %s

## GPRC
c.lw  ra, 4(sp) # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
c.sw  sp, 4(sp) # CHECK: :[[@LINE]]:7: error: invalid operand for instruction

## GPRNoX0
c.lwsp  x0, 4(sp) # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
c.lwsp  zero, 4(sp) # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

# Out of range immediates

## uimm8_lsb00
c.lwsp  ra, 256(sp) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 4 bytes in the range [0, 252]
c.swsp  ra, -4(sp) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 4 bytes in the range [0, 252]
## uimm7_lsb00
c.lw  s0, -4(sp) # CHECK: :[[@LINE]]:11: error: immediate must be a multiple of 4 bytes in the range [0, 124]
c.sw  s0, 128(sp) # CHECK: :[[@LINE]]:11: error: immediate must be a multiple of 4 bytes in the range [0, 124]
