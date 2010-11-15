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