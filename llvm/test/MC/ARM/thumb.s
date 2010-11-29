@ RUN: llvm-mc -triple thumbv6-apple-darwin -show-encoding < %s | FileCheck %s
        .code 16

@ CHECK: cmp	r1, r2               @ encoding: [0x91,0x42]
        cmp     r1, r2

@ CHECK: pop    {r1, r2, r4}         @ encoding: [0x16,0xbc]
        pop     {r1, r2, r4}

@ CHECK: trap                        @ encoding: [0xfe,0xde]
        trap

@ CHECK: blx	r9                   @ encoding: [0xc8,0x47]
	blx	r9

@ CHECK: rev	r2, r3               @ encoding: [0x1a,0xba]
@ CHECK: rev16	r3, r4               @ encoding: [0x63,0xba]
@ CHECK: revsh	r5, r6               @ encoding: [0xf5,0xba]
        rev     r2, r3
        rev16   r3, r4
        revsh   r5, r6

@ CHECK: sxtb	r2, r3               @ encoding: [0x5a,0xb2]
@ CHECK: sxth	r2, r3               @ encoding: [0x1a,0xb2]
	sxtb	r2, r3
	sxth	r2, r3
