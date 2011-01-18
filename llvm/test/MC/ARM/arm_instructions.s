@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding %s | FileCheck %s

@ CHECK: nop
@ CHECK: encoding: [0x00,0xf0,0x20,0xe3]
        nop

@ CHECK: nopeq
@ CHECK: encoding: [0x00,0xf0,0x20,0x03]
        nopeq

@ CHECK: trap
@ CHECK: encoding: [0xfe,0xde,0xff,0xe7]
        trap

@ CHECK: bx	lr
@ CHECK: encoding: [0x1e,0xff,0x2f,0xe1]
        bx lr

@ CHECK: vqdmull.s32	q8, d17, d16
@ CHECK: encoding: [0xa0,0x0d,0xe1,0xf2]
        vqdmull.s32     q8, d17, d16

@ CHECK: ldmia r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x92,0xe8]
@ CHECK: ldmib r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x92,0xe9]
@ CHECK: ldmda r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x12,0xe8]
@ CHECK: ldmdb r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x12,0xe9]
        ldmia     r2, {r1,r3-r6,sp}
        ldmib     r2, {r1,r3-r6,sp}
        ldmda     r2, {r1,r3-r6,sp}
        ldmdb     r2, {r1,r3-r6,sp}

@ CHECK: stmia r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x82,0xe8]
@ CHECK: stmib r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x82,0xe9]
@ CHECK: stmda r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x02,0xe8]
@ CHECK: stmdb r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x02,0xe9]
        stmia     r2, {r1,r3-r6,sp}
        stmib     r2, {r1,r3-r6,sp}
        stmda     r2, {r1,r3-r6,sp}
        stmdb     r2, {r1,r3-r6,sp}

@ CHECK: ldmia r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xb2,0xe8]
@ CHECK: ldmib r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xb2,0xe9]
@ CHECK: ldmda r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x32,0xe8]
@ CHECK: ldmdb r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x32,0xe9]
        ldmia     r2!, {r1,r3-r6,sp}
        ldmib     r2!, {r1,r3-r6,sp}
        ldmda     r2!, {r1,r3-r6,sp}
        ldmdb     r2!, {r1,r3-r6,sp}

@ CHECK: stmia r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xa2,0xe8]
@ CHECK: stmib r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xa2,0xe9]
@ CHECK: stmda r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x22,0xe8]
@ CHECK: stmdb r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x22,0xe9]
        stmia     r2!, {r1,r3-r6,sp}
        stmib     r2!, {r1,r3-r6,sp}
        stmda     r2!, {r1,r3-r6,sp}
        stmdb     r2!, {r1,r3-r6,sp}

@ CHECK: and	r1, r2, r3 @ encoding: [0x03,0x10,0x02,0xe0]
        and r1,r2,r3

@ FIXME: This is wrong, we are dropping the 's' for now.
@ CHECK-FIXME: ands	r1, r2, r3 @ encoding: [0x03,0x10,0x12,0xe0]
        ands r1,r2,r3

@ CHECK: eor	r1, r2, r3 @ encoding: [0x03,0x10,0x22,0xe0]
        eor r1,r2,r3

@ FIXME: This is wrong, we are dropping the 's' for now.
@ CHECK-FIXME: eors	r1, r2, r3 @ encoding: [0x03,0x10,0x32,0xe0]
        eors r1,r2,r3

@ CHECK: sub	r1, r2, r3 @ encoding: [0x03,0x10,0x42,0xe0]
        sub r1,r2,r3

@ FIXME: This is wrong, we are dropping the 's' for now.
@ CHECK-FIXME: subs	r1, r2, r3 @ encoding: [0x03,0x10,0x52,0xe0]
        subs r1,r2,r3

@ CHECK: add	r1, r2, r3 @ encoding: [0x03,0x10,0x82,0xe0]
        add r1,r2,r3

@ FIXME: This is wrong, we are dropping the 's' for now.
@ CHECK-FIXME: adds	r1, r2, r3 @ encoding: [0x03,0x10,0x92,0xe0]
        adds r1,r2,r3

@ CHECK: adc	r1, r2, r3 @ encoding: [0x03,0x10,0xa2,0xe0]
        adc r1,r2,r3

@ CHECK: sbc	r1, r2, r3 @ encoding: [0x03,0x10,0xc2,0xe0]
        sbc r1,r2,r3

@ CHECK: orr	r1, r2, r3 @ encoding: [0x03,0x10,0x82,0xe1]
        orr r1,r2,r3

@ FIXME: This is wrong, we are dropping the 's' for now.
@ CHECK-FIXME: orrs	r1, r2, r3 @ encoding: [0x03,0x10,0x92,0xe1]
        orrs r1,r2,r3

@ CHECK: bic	r1, r2, r3 @ encoding: [0x03,0x10,0xc2,0xe1]
        bic r1,r2,r3

@ FIXME: This is wrong, we are dropping the 's' for now.
@ CHECK-FIXME: bics	r1, r2, r3 @ encoding: [0x03,0x10,0xd2,0xe1]
        bics r1,r2,r3

@ CHECK: mov	r1, r2 @ encoding: [0x02,0x10,0xa0,0xe1]
        mov r1,r2

@ CHECK: mvn	r1, r2 @ encoding: [0x02,0x10,0xe0,0xe1]
        mvn r1,r2

@ FIXME: This is wrong, we are dropping the 's' for now.
@ CHECK-FIXME: mvns	r1, r2 @ encoding: [0x02,0x10,0xf0,0xe1]
        mvns r1,r2

@ CHECK: rsb	r1, r2, r3 @ encoding: [0x03,0x10,0x62,0xe0]
        rsb r1,r2,r3

@ CHECK: rsc	r1, r2, r3 @ encoding: [0x03,0x10,0xe2,0xe0]
        rsc r1,r2,r3

@ FIXME: This is broken, CCOut operands don't work correctly when their presence
@ may depend on flags.
@ CHECK-FIXME:: mlas	r1, r2, r3, r4 @ encoding: [0x92,0x43,0x31,0xe0]
@        mlas r1,r2,r3,r4

@ CHECK: bfi  r0, r0, #5, #7 @ encoding: [0x90,0x02,0xcb,0xe7]
        bfi  r0, r0, #5, #7

@ CHECK: bkpt  #10 @ encoding: [0x7a,0x00,0x20,0xe1]
        bkpt  #10

@ CHECK: isb @ encoding: [0x6f,0xf0,0x7f,0xf5]
        isb
