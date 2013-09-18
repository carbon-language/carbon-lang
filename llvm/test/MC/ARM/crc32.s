@ RUN: llvm-mc -triple=armv8 -show-encoding < %s | FileCheck %s
@ RUN: not llvm-mc -triple=armv7 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V7
        crc32b  r0, r1, r2
        crc32h  r0, r1, r2
        crc32w  r0, r1, r2

@ CHECK:  crc32b    r0, r1, r2              @ encoding: [0x42,0x00,0x01,0xe1]
@ CHECK:  crc32h    r0, r1, r2              @ encoding: [0x42,0x00,0x21,0xe1]
@ CHECK:  crc32w    r0, r1, r2              @ encoding: [0x42,0x00,0x41,0xe1]
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8

        crc32cb  r0, r1, r2
        crc32ch  r0, r1, r2
        crc32cw  r0, r1, r2

@ CHECK:  crc32cb   r0, r1, r2              @ encoding: [0x42,0x02,0x01,0xe1]
@ CHECK:  crc32ch   r0, r1, r2              @ encoding: [0x42,0x02,0x21,0xe1]
@ CHECK:  crc32cw   r0, r1, r2              @ encoding: [0x42,0x02,0x41,0xe1]
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8
