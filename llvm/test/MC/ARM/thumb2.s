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

@ CHECK: smlabt	r0, r1, r2, r0          @ encoding: [0xc1,0x02,0x00,0xe1]
  smlabt	r0, r1, r2, r0

@ CHECK: clz	r0, r0                  @ encoding: [0x10,0x0f,0x6f,0xe1]
  clz	r0, r0
@ CHECK: rev	r0, r0                  @ encoding: [0x30,0x0f,0xbf,0xe6]
  rev	r0, r0
@ CHECK: revsh	r0, r0                  @ encoding: [0xb0,0x0f,0xff,0xe6]
  revsh	r0, r0

@ CHECK: pkhbt	r0, r0, r1, lsl #16     @ encoding: [0x11,0x08,0x80,0xe6]
  pkhbt	r0, r0, r1, lsl #16
@ CHECK: pkhbt	r0, r0, r1, lsl #12     @ encoding: [0x11,0x06,0x80,0xe6]
  pkhbt	r0, r0, r1, lsl #16
@ CHECK: pkhbt	r0, r0, r1, lsl #18     @ encoding: [0x11,0x09,0x80,0xe6]
  pkhbt	r0, r0, r1, lsl #18
@ CHECK: pkhbt	r0, r0, r1              @ encoding: [0x11,0x00,0x80,0xe6]
  pkhbt	r0, r0, r1
@ CHECK: pkhtb	r0, r0, r1, asr #16     @ encoding: [0x51,0x08,0x80,0xe6]
  pkhtb	r0, r0, r1, asr #16
@ CHECK: pkhtb	r0, r0, r1, asr #12     @ encoding: [0x51,0x06,0x80,0xe6]
  pkhtb	r0, r0, r1, asr #12
@ CHECK: pkhtb	r0, r0, r1, asr #18     @ encoding: [0x51,0x09,0x80,0xe6]
  pkhtb	r0, r0, r1, asr #18
@ CHECK: pkhtb	r0, r0, r1, asr #22     @ encoding: [0x51,0x0b,0x80,0xe6]
  pkhtb	r0, r0, r1, asr #22
