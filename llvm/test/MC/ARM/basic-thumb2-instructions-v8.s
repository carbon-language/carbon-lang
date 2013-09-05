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
