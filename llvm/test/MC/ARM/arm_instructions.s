@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding %s | FileCheck %s

@ CHECK: trap
@ CHECK: encoding: [0xfe,0xde,0xff,0xe7]
        trap

@ CHECK: bx	lr
@ CHECK: encoding: [0x1e,0xff,0x2f,0xe1]
        bx lr

@ CHECK: vqdmull.s32	q8, d17, d16
@ CHECK: encoding: [0xa0,0x0d,0xe1,0xf2]
        vqdmull.s32     q8, d17, d16

@ CHECK: and	r1, r2, r3 @ encoding: [0x03,0x10,0x02,0xe0]
        and r1,r2,r3

@ CHECK: ands	r1, r2, r3 @ encoding: [0x03,0x10,0x12,0xe0]
        ands r1,r2,r3

@ CHECK: eor	r1, r2, r3 @ encoding: [0x03,0x10,0x22,0xe0]
        eor r1,r2,r3

@ CHECK: eors	r1, r2, r3 @ encoding: [0x03,0x10,0x32,0xe0]
        eors r1,r2,r3

@ CHECK: sub	r1, r2, r3 @ encoding: [0x03,0x10,0x42,0xe0]
        sub r1,r2,r3

@ CHECK: subs	r1, r2, r3 @ encoding: [0x03,0x10,0x52,0xe0]
        subs r1,r2,r3

@ CHECK: add	r1, r2, r3 @ encoding: [0x03,0x10,0x82,0xe0]
        add r1,r2,r3

@ CHECK: adds	r1, r2, r3 @ encoding: [0x03,0x10,0x92,0xe0]
        adds r1,r2,r3

@ CHECK: adc	r1, r2, r3 @ encoding: [0x03,0x10,0xa2,0xe0]
        adc r1,r2,r3

@ CHECK: sbc	r1, r2, r3 @ encoding: [0x03,0x10,0xc2,0xe0]
        sbc r1,r2,r3

@ CHECK: bic	r1, r2, r3 @ encoding: [0x03,0x10,0xc2,0xe1]
        bic r1,r2,r3

@ CHECK: bics	r1, r2, r3 @ encoding: [0x03,0x10,0xd2,0xe1]
        bics r1,r2,r3

@ CHECK: mov	r1, r2 @ encoding: [0x02,0x10,0xa0,0xe1]
        mov r1,r2

@ CHECK: mvn	r1, r2 @ encoding: [0x02,0x10,0xe0,0xe1]
        mvn r1,r2

@ CHECK: mvns	r1, r2 @ encoding: [0x02,0x10,0xf0,0xe1]
        mvns r1,r2

@ CHECK: rsb	r1, r2, r3 @ encoding: [0x03,0x10,0x62,0xe0]
        rsb r1,r2,r3

@ CHECK: rsc	r1, r2, r3 @ encoding: [0x03,0x10,0xe2,0xe0]
        rsc r1,r2,r3

@ CHECK: bfi  r0, r0, #5, #7 @ encoding: [0x90,0x02,0xcb,0xe7]
        bfi  r0, r0, #5, #7

@ CHECK: bkpt  #10 @ encoding: [0x7a,0x00,0x20,0xe1]
        bkpt  #10

@ CHECK: cdp  p7, #1, c1, c1, c1, #4 @ encoding: [0x81,0x17,0x11,0xee]
        cdp  p7, #1, c1, c1, c1, #4
@ CHECK: cdp2  p7, #1, c1, c1, c1, #4 @ encoding: [0x81,0x17,0x11,0xfe]
        cdp2  p7, #1, c1, c1, c1, #4

@ CHECK: wfe @ encoding: [0x02,0xf0,0x20,0xe3]
        wfe

@ CHECK: wfi @ encoding: [0x03,0xf0,0x20,0xe3]
        wfi

@ CHECK: yield @ encoding: [0x01,0xf0,0x20,0xe3]
        yield

@ CHECK: cpsie  aif @ encoding: [0xc0,0x01,0x08,0xf1]
        cpsie  aif

@ CHECK: cps  #15 @ encoding: [0x0f,0x00,0x02,0xf1]
        cps  #15

@ CHECK: cpsie  if, #10 @ encoding: [0xca,0x00,0x0a,0xf1]
        cpsie  if, #10

@ CHECK: add	r1, r2, r3, lsl r4      @ encoding: [0x13,0x14,0x82,0xe0]
  add r1, r2, r3, lsl r4

@ CHECK: strexb  r0, r1, [r2] @ encoding: [0x91,0x0f,0xc2,0xe1]
        strexb  r0, r1, [r2]

@ CHECK: strexh  r0, r1, [r2] @ encoding: [0x91,0x0f,0xe2,0xe1]
        strexh  r0, r1, [r2]

@ CHECK: strex  r0, r1, [r2] @ encoding: [0x91,0x0f,0x82,0xe1]
        strex  r0, r1, [r2]

@ CHECK: strexd  r0, r2, r3, [r1] @ encoding: [0x92,0x0f,0xa1,0xe1]
        strexd  r0, r2, r3, [r1]

@ CHECK: ldrexb  r0, [r0] @ encoding: [0x9f,0x0f,0xd0,0xe1]
        ldrexb  r0, [r0]

@ CHECK: ldrexh  r0, [r0] @ encoding: [0x9f,0x0f,0xf0,0xe1]
        ldrexh  r0, [r0]

@ CHECK: ldrex  r0, [r0] @ encoding: [0x9f,0x0f,0x90,0xe1]
        ldrex  r0, [r0]

@ CHECK: ldrexd  r0, r1, [r0] @ encoding: [0x9f,0x0f,0xb0,0xe1]
        ldrexd  r0, r1, [r0]

@ CHECK: ssat16  r0, #7, r0 @ encoding: [0x30,0x0f,0xa6,0xe6]
        ssat16  r0, #7, r0

