@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s
@ XFAIL: *
.code 16

@ CHECK: adc	r1, r1, #171            @ encoding: [0xab,0x01,0x41,0xf1]
  adc	r1, r1, #171
@ CHECK: adc	r1, r1, #1179666        @ encoding: [0x12,0x11,0x41,0xf1]
  adc	r1, r1, #1179666
@ CHECK: adc	r1, r1, #872428544      @ encoding: [0x34,0x21,0x41,0xf1]
  adc	r1, r1, #872428544
@ CHECK: adc	r1, r1, #1448498774     @ encoding: [0x56,0x31,0x41,0xf1]
  adc	r1, r1, #1448498774
@ CHECK: adc	r1, r1, #66846720       @ encoding: [0x7f,0x71,0x41,0xf1]
  adc	r1, r1, #66846720

@ CHECK: mvn	r0, #187                @ encoding: [0xbb,0x00,0x6f,0xf0]
  mvn	r0, #187
@ CHECK: mvn	r0, #11141290           @ encoding: [0xaa,0x10,0x6f,0xf0]
  mvn	r0, #11141290
@ CHECK: mvn	r0, #-872363008         @ encoding: [0xcc,0x20,0x6f,0xf0]
  mvn	r0, #-872363008
@ CHECK: mvn	r0, #1114112            @ encoding: [0x88,0x10,0x6f,0xf4]
  mvn	r0, #1114112

@ CHECK: cmp.w	r0, #11141290           @ encoding: [0xaa,0x1f,0xb0,0xf1]
  cmp.w	r0, #11141290
@ CHECK: cmp.w	r0, #-872363008         @ encoding: [0xcc,0x2f,0xb0,0xf1]
  cmp.w	r0, #-872363008
@ CHECK: cmp.w	r0, #-572662307         @ encoding: [0xdd,0x3f,0xb0,0xf1]
  cmp.w	r0, #-572662307
@ CHECK: cmp.w	r0, #1114112            @ encoding: [0x88,0x1f,0xb0,0xf5]
  cmp.w	r0, #1114112
@ CHECK: cmp.w	r0, r1, lsl #5          @ encoding: [0x41,0x1f,0xb0,0xeb]
  cmp.w	r0, r1, lsl #5

@ CHECK: sxtab	r0, r1, r0              @ encoding: [0x80,0xf0,0x41,0xfa]
  sxtab	r0, r1, r0              @ encoding: [0x80,0xf0,0x41,0xfa]
  
@ CHECK: movw	r0, #65535              @ encoding: [0xff,0x70,0x4f,0xf6]
  movw	r0, #65535
@ CHECK: movw	r1, #43777              @ encoding: [0x01,0x31,0x4a,0xf6]
  movw	r1, #43777
@ CHECK: movt	r1, #427                @ encoding: [0xab,0x11,0xc0,0xf2]
  movt	r1, #427
@ CHECK: movw	r1, #43792              @ encoding: [0x10,0x31,0x4a,0xf6]
  movw	r1, #43792
@ CHECK: movt	r1, #4267               @ encoding: [0xab,0x01,0xc0,0xf2]
  movt	r1, #4267
@ CHECK: mov.w	r0, #66846720           @ encoding: [0x7f,0x70,0x4f,0xf0]
  mov.w	r0, #66846720

@ CHECK: rrx	r0, r0                  @ encoding: [0x30,0x00,0x4f,0xea]
  rrx	r0, r0

@ CHECK: bfc	r0, #4, #20             @ encoding: [0x17,0x10,0x6f,0xf3]
  bfc	r0, #4, #20
@ CHECK: bfc	r0, #0, #23             @ encoding: [0x16,0x00,0x6f,0xf3]
  bfc	r0, #0, #23
@ CHECK: bfc	r0, #12, #20            @ encoding: [0x1f,0x30,0x6f,0xf3]
  bfc	r0, #12, #20

@ CHECK: sbfx	r0, r0, #7, #11         @ encoding: [0xca,0x10,0x40,0xf3]
  sbfx	r0, r0, #7, #11
@ CHECK: ubfx	r0, r0, #7, #11         @ encoding: [0xca,0x10,0xc0,0xf3]
  ubfx	r0, r0, #7, #11

@ CHECK: mla	r0, r0, r1, r2          @ encoding: [0x01,0x20,0x00,0xfb]
  mla	r0, r0, r1, r2
@ CHECK: mls	r0, r0, r1, r2          @ encoding: [0x11,0x20,0x00,0xfb]
  mls	r0, r0, r1, r2

@ CHECK: smlabt	r0, r1, r2, r0          @ encoding: [0x12,0x00,0x11,0xfb]
  smlabt	r0, r1, r2, r0

@ CHECK: clz	r0, r0                  @ encoding: [0x80,0xf0,0xb0,0xfa]
  clz	r0, r0

@ CHECK: pkhbt	r0, r0, r1, lsl #16     @ encoding: [0x01,0x40,0xc0,0xea]
  pkhbt	r0, r0, r1, lsl #16
@ CHECK: pkhbt	r0, r0, r1, lsl #12     @ encoding: [0x01,0x30,0xc0,0xea]
  pkhbt	r0, r0, r1, lsl #16
@ CHECK: pkhbt	r0, r0, r1, lsl #18     @ encoding: [0x81,0x40,0xc0,0xea]
  pkhbt	r0, r0, r1, lsl #18
@ CHECK: pkhbt	r0, r0, r1              @ encoding: [0x01,0x00,0xc0,0xea]
  pkhbt	r0, r0, r1
@ CHECK: pkhtb	r0, r0, r1, asr #16     @ encoding: [0x21,0x40,0xc0,0xea]
  pkhtb	r0, r0, r1, asr #16
@ CHECK: pkhtb	r0, r0, r1, asr #12     @ encoding: [0x21,0x30,0xc0,0xea]
  pkhtb	r0, r0, r1, asr #12
@ CHECK: pkhtb	r0, r0, r1, asr #18     @ encoding: [0xa1,0x40,0xc0,0xea]
  pkhtb	r0, r0, r1, asr #18
@ CHECK: pkhtb	r0, r0, r1, asr #22     @ encoding: [0xa1,0x50,0xc0,0xea]
  pkhtb	r0, r0, r1, asr #22

@ CHECK: dmb	st                      @ encoding: [0x5e,0x8f,0xbf,0xf3]
  dmb	st
@ CHECK: dmb	sy                      @ encoding: [0x5f,0x8f,0xbf,0xf3]
  dmb	sy
@ CHECK: dmb	ishst                   @ encoding: [0x5a,0x8f,0xbf,0xf3]
  dmb	ishst
@ CHECK: dmb	ish                     @ encoding: [0x5b,0x8f,0xbf,0xf3]
  dmb	ish

@ CHECK: str.w	r0, [r1, #4092]         @ encoding: [0xfc,0x0f,0xc1,0xf8]
  str.w	r0, [r1, #4092]
@ CHECK: str	r0, [r1, #-128]         @ encoding: [0x80,0x0c,0x41,0xf8]
  str	r0, [r1, #-128]
@ CHECK: str.w	r0, [r1, r2, lsl #2]    @ encoding: [0x22,0x00,0x41,0xf8
  str.w	r0, [r1, r2, lsl #2]

@ CHECK: ldr.w	r0, [r0, #4092]         @ encoding: [0xfc,0x0f,0xd0,0xf8]
  ldr.w	r0, [r0, #4092]
@ CHECK: ldr	r0, [r0, #-128]         @ encoding: [0x80,0x0c,0x50,0xf8]
  ldr	r0, [r0, #-128]
@ CHECK: ldr.w	r0, [r0, r1, lsl #2]    @ encoding: [0x21,0x00,0x50,0xf8]
  ldr.w	r0, [r0, r1, lsl #2]

@ CHECK: str	r1, [r0, #16]!          @ encoding: [0x10,0x1f,0x40,0xf8]
  str	r1, [r0, #16]!
@ CHECK: strh	r1, [r0, #8]!           @ encoding: [0x08,0x1f,0x20,0xf8]
  strh	r1, [r0, #8]!
@ CHECK: strh	r2, [r0], #-4           @ encoding: [0x04,0x29,0x20,0xf8]
  strh	r2, [r0], #-4
@ CHECK: str	r2, [r0], #-4           @ encoding: [0x04,0x29,0x40,0xf8]
  str	r2, [r0], #-4

@ CHECK: ldr	r2, [r0, #16]!          @ encoding: [0x10,0x2f,0x50,0xf8]
  ldr	r2, [r0, #16]!
@ CHECK: ldr	r2, [r0, #-64]!         @ encoding: [0x40,0x2d,0x50,0xf8]
  ldr	r2, [r0, #-64]!
@ CHECK: ldrsb	r2, [r0, #4]!           @ encoding: [0x04,0x2f,0x10,0xf9]
  ldrsb	r2, [r0, #4]!

@ CHECK: strb.w	r0, [r1, #4092]         @ encoding: [0xfc,0x0f,0x81,0xf8]
  strb.w	r0, [r1, #4092]
@ CHECK: strb	r0, [r1, #-128]         @ encoding: [0x80,0x0c,0x01,0xf8]
  strb	r0, [r1, #-128]
@ CHECK: strb.w	r0, [r1, r2, lsl #2]    @ encoding: [0x22,0x00,0x01,0xf8]
  strb.w	r0, [r1, r2, lsl #2]
@ CHECK: strh.w	r0, [r1, #4092]         @ encoding: [0xfc,0x0f,0xa1,0xf8]
  strh.w	r0, [r1, #4092]
@ CHECK: strh	r0, [r1, #-128]         @ encoding: [0x80,0x0c,0x21,0xf8]
  strh	r0, [r1, #-128]
@ CHECK: strh	r0, [r1, #-128]         @ encoding: [0x80,0x0c,0x21,0xf8]
  strh	r0, [r1, #-128]
@ CHECK: strh.w	r0, [r1, r2, lsl #2]    @ encoding: [0x22,0x00,0x21,0xf8]
  strh.w	r0, [r1, r2, lsl #2]

@ CHECK: ldrb	r0, [r0, #-1]           @ encoding: [0x01,0x0c,0x10,0xf8]
  ldrb	r0, [r0, #-1]
@ CHECK: ldrb	r0, [r0, #-128]         @ encoding: [0x80,0x0c,0x10,0xf8]
  ldrb	r0, [r0, #-128]
@ CHECK: ldrb.w	r0, [r0, r1, lsl #2]    @ encoding: [0x21,0x00,0x10,0xf8]
  ldrb.w	r0, [r0, r1, lsl #2]
@ CHECK: ldrh.w	r0, [r0, #2046]         @ encoding: [0xfe,0x07,0xb0,0xf8]
  ldrh.w	r0, [r0, #2046]
@ CHECK: ldrh	r0, [r0, #-128]         @ encoding: [0x80,0x0c,0x30,0xf8]
  ldrh	r0, [r0, #-128]
@ CHECK: ldrh.w	r0, [r0, r1, lsl #2]    @ encoding: [0x21,0x00,0x30,0xf8]
  ldrh.w	r0, [r0, r1, lsl #2]
@ CHECK: ldrsb.w	r0, [r0]                @ encoding: [0x00,0x00,0x90,0xf9]
  ldrsb.w	r0, [r0]
@ CHECK: ldrsh.w	r0, [r0]                @ encoding: [0x00,0x00,0xb0,0xf9]
  ldrsh.w	r0, [r0]
@ CHECK: bfi  r0, r0, #5, #7 @ encoding: [0x60,0xf3,0x4b,0x10]
  bfi  r0, r0, #5, #7
@ CHECK: isb @ encoding: [0xbf,0xf3,0x6f,0x8f]
  isb
@ CHECK: mrs  r0, cpsr @ encoding: [0xef,0xf3,0x00,0x80]
  mrs  r0, cpsr
@ CHECK: vmrs  r0, fpscr @ encoding: [0xf1,0xee,0x10,0x0a]
  vmrs  r0, fpscr
@ CHECK: vmrs  r0, fpexc @ encoding: [0xf8,0xee,0x10,0x0a]
  vmrs  r0, fpexc
@ CHECK: vmrs  r0, fpsid @ encoding: [0xf0,0xee,0x10,0x0a]
  vmrs  r0, fpsid

@ CHECK: vmsr  fpscr, r0 @ encoding: [0xe1,0xee,0x10,0x0a]
  vmsr  fpscr, r0
@ CHECK: vmsr  fpexc, r0 @ encoding: [0xe8,0xee,0x10,0x0a]
  vmsr  fpexc, r0
@ CHECK: vmsr  fpsid, r0 @ encoding: [0xe0,0xee,0x10,0x0a]
  vmsr  fpsid, r0

@ CHECK: mcr2  p7, #1, r5, c1, c1, #4 @ encoding: [0x21,0xfe,0x91,0x57]
        mcr2  p7, #1, r5, c1, c1, #4

@ CHECK: mrc2  p14, #0, r1, c1, c2, #4 @ encoding: [0x11,0xfe,0x92,0x1e]
        mrc2  p14, #0, r1, c1, c2, #4

@ CHECK: mcrr2  p7, #1, r5, r4, c1 @ encoding: [0x44,0xfc,0x11,0x57]
        mcrr2  p7, #1, r5, r4, c1

@ CHECK: mrrc2  p7, #1, r5, r4, c1 @ encoding: [0x54,0xfc,0x11,0x57]
        mrrc2  p7, #1, r5, r4, c1

@ CHECK: cdp2  p7, #1, c1, c1, c1, #4 @ encoding: [0x11,0xfe,0x81,0x17]
        cdp2  p7, #1, c1, c1, c1, #4

@ CHECK: clrex @ encoding: [0xbf,0xf3,0x2f,0x8f]
        clrex

@ CHECK: clz  r9, r0 @ encoding: [0xb0,0xfa,0x80,0xf9]
        clz  r9, r0

@ CHECK: qadd  r1, r2, r3 @ encoding: [0x83,0xfa,0x82,0xf1]
        qadd  r1, r2, r3

@ CHECK: qsub  r1, r2, r3 @ encoding: [0x83,0xfa,0xa2,0xf1]
        qsub  r1, r2, r3

@ CHECK: qdadd  r1, r2, r3 @ encoding: [0x83,0xfa,0x92,0xf1]
        qdadd  r1, r2, r3

@ CHECK: qdsub  r1, r2, r3 @ encoding: [0x83,0xfa,0xb2,0xf1]
        qdsub  r1, r2, r3

@ CHECK: nop.w @ encoding: [0xaf,0xf3,0x00,0x80]
        nop.w

@ CHECK: yield.w @ encoding: [0xaf,0xf3,0x01,0x80]
        yield.w

@ CHECK: wfe.w @ encoding: [0xaf,0xf3,0x02,0x80]
        wfe.w

@ CHECK: wfi.w @ encoding: [0xaf,0xf3,0x03,0x80]
        wfi.w

