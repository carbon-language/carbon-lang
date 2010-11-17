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

