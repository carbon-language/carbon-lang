@ RUN: llvm-mc -triple=thumbv7-apple-darwin -mcpu=cortex-a8 -mattr=+mp -show-encoding < %s | FileCheck %s

@------------------------------------------------------------------------------
@ PLD(literal)
@------------------------------------------------------------------------------
        pldw [pc,#-4095]
@ CHECK: pldw [pc, #-4095]            @ encoding: [0x3f,0xf8,0xff,0xff]
