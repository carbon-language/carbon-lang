@ RUN: not llvm-mc -triple=armv7-apple-darwin -mcpu=cortex-a8 < %s 2>&1 | FileCheck %s

hint #5
hint #100

@ CHECK: error: immediate operand must be in the range [0,4]
@ CHECK: error: immediate operand must be in the range [0,4]
