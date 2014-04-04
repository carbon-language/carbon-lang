@ PR18921
@ RUN: llvm-mc -triple=armv7-linux-gnueabi -show-encoding < %s | FileCheck %s
.text
.thumb

@ CHECK: .code	16

@ CHECK: ldrd	r0, r1, [r10, #512]!    @ encoding: [0xfa,0xe9,0x80,0x01]
@ CHECK: ldrd	r0, r1, [r10], #512     @ encoding: [0xfa,0xe8,0x80,0x01]
@ CHECK: ldrd	r0, r1, [r10, #512]     @ encoding: [0xda,0xe9,0x80,0x01]
        ldrd    r0, [r10, #512]!
        ldrd    r0, [r10], #512
        ldrd    r0, [r10, #512]

@ CHECK: strd	r0, r1, [r10, #512]!    @ encoding: [0xea,0xe9,0x80,0x01]
@ CHECK: strd	r0, r1, [r10], #512     @ encoding: [0xea,0xe8,0x80,0x01]
@ CHECK: strd	r0, r1, [r10, #512]     @ encoding: [0xca,0xe9,0x80,0x01]
        strd    r0, [r10, #512]!
        strd    r0, [r10], #512
        strd    r0, [r10, #512]
