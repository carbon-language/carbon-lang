@ RUN: not llvm-mc -triple armv7a--none-eabi < %s |& FileCheck %s
@ RUN: not llvm-mc -triple thumbv7a--none-eabi < %s |& FileCheck %s

  msr apsr_c, r0
@ CHECK: invalid operand for instruction
  msr cpsr_w
@ CHECK: invalid operand for instruction
  msr cpsr_cc
@ CHECK: invalid operand for instruction
  msr xpsr_c
@ CHECK: invalid operand for instruction
