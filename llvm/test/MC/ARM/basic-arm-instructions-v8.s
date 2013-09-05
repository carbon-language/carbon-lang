@ New ARMv8 A32 encodings

@ RUN: llvm-mc -triple armv8 -show-encoding < %s | FileCheck %s --check-prefix=CHECK-V8
@ RUN: not llvm-mc -triple armv7 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V7

@ HLT
        hlt  #0
        hlt  #65535
@ CHECK-V8: hlt  #0                       @ encoding: [0x70,0x00,0x00,0xe1]
@ CHECK-V8: hlt  #65535                   @ encoding: [0x7f,0xff,0x0f,0xe1]
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8

@ AL condition code allowable
        hltal  #0
@ CHECK-V8: hlt  #0                       @ encoding: [0x70,0x00,0x00,0xe1]
@ CHECK-V7: error: instruction requires: armv8
