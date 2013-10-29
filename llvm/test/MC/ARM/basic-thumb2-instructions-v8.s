@ New ARMv8 T32 encodings

@ RUN: llvm-mc -triple thumbv8 -show-encoding < %s | FileCheck %s --check-prefix=CHECK-V8
@ RUN: not llvm-mc -triple thumbv7 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V7

@ HLT
        hlt  #0
        hlt  #63
@ CHECK-V8: hlt  #0                       @ encoding: [0x80,0xba]
@ CHECK-V8: hlt  #63                      @ encoding: [0xbf,0xba]
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8

@ In IT block
        it pl
        hlt #24

@ CHECK-V8: it pl                         @ encoding: [0x58,0xbf]
@ CHECK-V8: hlt #24                       @ encoding: [0x98,0xba]
@ CHECK-V7: error: instruction requires: armv8

@ Can accept AL condition code
        hltal #24
@ CHECK-V8: hlt #24                       @ encoding: [0x98,0xba]
@ CHECK-V7: error: instruction requires: armv8

@ DCPS{1,2,3}
        dcps1
        dcps2
        dcps3
@ CHECK-V8: dcps1                         @ encoding: [0x8f,0xf7,0x01,0x80]
@ CHECK-V8: dcps2                         @ encoding: [0x8f,0xf7,0x02,0x80]
@ CHECK-V8: dcps3                         @ encoding: [0x8f,0xf7,0x03,0x80]
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8

@------------------------------------------------------------------------------
@ DMB (v8 barriers)
@------------------------------------------------------------------------------
        dmb ishld
        dmb oshld
        dmb nshld
        dmb ld

@ CHECK-V8: dmb ishld @ encoding: [0xbf,0xf3,0x59,0x8f]
@ CHECK-V8: dmb oshld @ encoding: [0xbf,0xf3,0x51,0x8f]
@ CHECK-V8: dmb nshld @ encoding: [0xbf,0xf3,0x55,0x8f]
@ CHECK-V8: dmb ld @ encoding: [0xbf,0xf3,0x5d,0x8f]
@ CHECK-V7: error: invalid operand for instruction
@ CHECK-V7: error: invalid operand for instruction
@ CHECK-V7: error: invalid operand for instruction
@ CHECK-V7: error: invalid operand for instruction

@------------------------------------------------------------------------------
@ DSB (v8 barriers)
@------------------------------------------------------------------------------
        dsb ishld
        dsb oshld
        dsb nshld
        dsb ld

@ CHECK-V8: dsb ishld @ encoding: [0xbf,0xf3,0x49,0x8f]
@ CHECK-V8: dsb oshld @ encoding: [0xbf,0xf3,0x41,0x8f]
@ CHECK-V8: dsb nshld @ encoding: [0xbf,0xf3,0x45,0x8f]
@ CHECK-V8: dsb ld @ encoding: [0xbf,0xf3,0x4d,0x8f]
@ CHECK-V7: error: invalid operand for instruction
@ CHECK-V7: error: invalid operand for instruction
@ CHECK-V7: error: invalid operand for instruction
@ CHECK-V7: error: invalid operand for instruction

@------------------------------------------------------------------------------
@ SEVL
@------------------------------------------------------------------------------
        sevl
        sevl.w
        it ge
        sevlge

@ CHECK-V8: sevl @ encoding: [0x50,0xbf]
@ CHECK-V8: sevl.w @ encoding: [0xaf,0xf3,0x05,0x80]
@ CHECK-V8: it ge @ encoding: [0xa8,0xbf]
@ CHECK-V8: sevlge @ encoding: [0x50,0xbf]
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error:
@ CHECK-V7: error: instruction requires: armv8
