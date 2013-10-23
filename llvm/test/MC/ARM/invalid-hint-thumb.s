@ RUN: not llvm-mc -triple=thumbv7-apple-darwin -mcpu=cortex-a8 < %s 2>&1 | FileCheck %s

hint #240
hint #1000

@ CHECK: error: immediate operand must be in the range [0,239]
@ CHECK: error: immediate operand must be in the range [0,239]

