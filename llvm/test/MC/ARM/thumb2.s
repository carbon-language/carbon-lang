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
