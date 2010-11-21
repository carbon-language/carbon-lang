@ RUN: llvm-mc -triple thumb-apple-darwin -show-encoding < %s | FileCheck %s
        .code 16

@ CHECK: cmp	r1, r2               @ encoding: [0x91,0x42]
        cmp     r1, r2

@ CHECK: pop    {r1, r2, r4}         @ encoding: [0x16,0xbc]
        pop     {r1, r2, r4}

@ CHECK: trap                        @ encoding: [0xfe,0xde]
        trap
