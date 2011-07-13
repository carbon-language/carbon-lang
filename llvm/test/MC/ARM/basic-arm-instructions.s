@ RUN: llvm-mc -triple=armv7-apple-darwin -show-encoding < %s | FileCheck %s
  .syntax unified
  .globl _func

@ Check that the assembler can handle the documented syntax from the ARM ARM.
@ For complex constructs like shifter operands, check more thoroughly for them
@ once then spot check that following instructions accept the form generally.
@ This gives us good coverage while keeping the overall size of the test
@ more reasonable.

_func:
@ CHECK: _func

@------------------------------------------------------------------------------
@ ADC (immediate)
@------------------------------------------------------------------------------
  adc r1, r2, #0xf
  adc r1, r2, #0xf0
  adc r1, r2, #0xf00
  adc r1, r2, #0xf000
  adc r1, r2, #0xf0000
  adc r1, r2, #0xf00000
  adc r1, r2, #0xf000000
  adc r1, r2, #0xf0000000
  adc r1, r2, #0xf000000f
  adcs r1, r2, #0xf00
  adcseq r1, r2, #0xf00
  adceq r1, r2, #0xf00

@ CHECK: adc	r1, r2, #15             @ encoding: [0x0f,0x10,0xa2,0xe2]
@ CHECK: adc	r1, r2, #240            @ encoding: [0xf0,0x10,0xa2,0xe2]
@ CHECK: adc	r1, r2, #3840           @ encoding: [0x0f,0x1c,0xa2,0xe2]
@ CHECK: adc	r1, r2, #61440          @ encoding: [0x0f,0x1a,0xa2,0xe2]
@ CHECK: adc	r1, r2, #983040         @ encoding: [0x0f,0x18,0xa2,0xe2]
@ CHECK: adc	r1, r2, #15728640       @ encoding: [0x0f,0x16,0xa2,0xe2]
@ CHECK: adc	r1, r2, #251658240      @ encoding: [0x0f,0x14,0xa2,0xe2]
@ CHECK: adc	r1, r2, #4026531840     @ encoding: [0x0f,0x12,0xa2,0xe2]
@ CHECK: adc	r1, r2, #4026531855     @ encoding: [0xff,0x12,0xa2,0xe2]

@ CHECK: adcs	r1, r2, #3840           @ encoding: [0x0f,0x1c,0xb2,0xe2]
@ CHECK: adcseq	r1, r2, #3840           @ encoding: [0x0f,0x1c,0xb2,0x02]
@ CHECK: adceq	r1, r2, #3840           @ encoding: [0x0f,0x1c,0xa2,0x02]

@------------------------------------------------------------------------------
@ ADC (register)
@ ADC (shifted register)
@------------------------------------------------------------------------------
  adc r4, r5, r6
  @ Constant shifts
  adc r4, r5, r6, lsl #1
  adc r4, r5, r6, lsl #31
  adc r4, r5, r6, lsr #1
  adc r4, r5, r6, lsr #31
  adc r4, r5, r6, lsr #32
  adc r4, r5, r6, asr #1
  adc r4, r5, r6, asr #31
  adc r4, r5, r6, asr #32
  adc r4, r5, r6, ror #1
  adc r4, r5, r6, ror #31

  @ Register shifts
  adc r6, r7, r8, lsl r9
  adc r6, r7, r8, lsr r9
  adc r6, r7, r8, asr r9
  adc r6, r7, r8, ror r9
  adc r4, r5, r6, rrx

  @ Destination register is optional
  adc r5, r6
  adc r4, r5, lsl #1
  adc r4, r5, lsl #31
  adc r4, r5, lsr #1
  adc r4, r5, lsr #31
  adc r4, r5, lsr #32
  adc r4, r5, asr #1
  adc r4, r5, asr #31
  adc r4, r5, asr #32
  adc r4, r5, ror #1
  adc r4, r5, ror #31
  adc r4, r5, rrx
  adc r6, r7, lsl r9
  adc r6, r7, lsr r9
  adc r6, r7, asr r9
  adc r6, r7, ror r9
  adc r4, r5, rrx

@ CHECK: adc	r4, r5, r6              @ encoding: [0x06,0x40,0xa5,0xe0]

@ CHECK: adc	r4, r5, r6, lsl #1      @ encoding: [0x86,0x40,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, lsl #31     @ encoding: [0x86,0x4f,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, lsr #1      @ encoding: [0xa6,0x40,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, lsr #31     @ encoding: [0xa6,0x4f,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, lsr #32     @ encoding: [0x26,0x40,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, asr #1      @ encoding: [0xc6,0x40,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, asr #31     @ encoding: [0xc6,0x4f,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, asr #32     @ encoding: [0x46,0x40,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, ror #1      @ encoding: [0xe6,0x40,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, ror #31     @ encoding: [0xe6,0x4f,0xa5,0xe0]

@ CHECK: adc	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0xa7,0xe0]
@ CHECK: adc	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0xa7,0xe0]
@ CHECK: adc	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0xa7,0xe0]
@ CHECK: adc	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0xa7,0xe0]
@ CHECK: adc	r4, r5, r6, rrx         @ encoding: [0x66,0x40,0xa5,0xe0]

@ CHECK: adc	r5, r5, r6              @ encoding: [0x06,0x50,0xa5,0xe0]
@ CHECK: adc	r4, r4, r5, lsl #1      @ encoding: [0x85,0x40,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, lsl #31     @ encoding: [0x85,0x4f,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, lsr #1      @ encoding: [0xa5,0x40,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, lsr #31     @ encoding: [0xa5,0x4f,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, lsr #32     @ encoding: [0x25,0x40,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, asr #1      @ encoding: [0xc5,0x40,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, asr #31     @ encoding: [0xc5,0x4f,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, asr #32     @ encoding: [0x45,0x40,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, ror #1      @ encoding: [0xe5,0x40,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, ror #31     @ encoding: [0xe5,0x4f,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0xa4,0xe0]
@ CHECK: adc	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0xa6,0xe0]
@ CHECK: adc	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0xa6,0xe0]
@ CHECK: adc	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0xa6,0xe0]
@ CHECK: adc	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0xa6,0xe0]
@ CHECK: adc	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0xa4,0xe0]


@------------------------------------------------------------------------------
@ ADD
@------------------------------------------------------------------------------
  add r4, r5, #0xf000
  add r4, r5, r6
  add r4, r5, r6, lsl #5
  add r4, r5, r6, lsr #5
  add r4, r5, r6, lsr #5
  add r4, r5, r6, asr #5
  add r4, r5, r6, ror #5
  add r6, r7, r8, lsl r9
  add r6, r7, r8, lsr r9
  add r6, r7, r8, asr r9
  add r6, r7, r8, ror r9
  add r4, r5, r6, rrx

  @ destination register is optional
  add r5, #0xf000
  add r4, r5
  add r4, r5, lsl #5
  add r4, r5, lsr #5
  add r4, r5, lsr #5
  add r4, r5, asr #5
  add r4, r5, ror #5
  add r6, r7, lsl r9
  add r6, r7, lsr r9
  add r6, r7, asr r9
  add r6, r7, ror r9
  add r4, r5, rrx

@ CHECK: add	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x85,0xe2]
@ CHECK: add	r4, r5, r6              @ encoding: [0x06,0x40,0x85,0xe0]
@ CHECK: add	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0x85,0xe0]
@ CHECK: add	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x85,0xe0]
@ CHECK: add	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x85,0xe0]
@ CHECK: add	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0x85,0xe0]
@ CHECK: add	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0x85,0xe0]
@ CHECK: add	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0x87,0xe0]
@ CHECK: add	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0x87,0xe0]
@ CHECK: add	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0x87,0xe0]
@ CHECK: add	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0x87,0xe0]
@ CHECK: add	r4, r5, r6, rrx         @ encoding: [0x66,0x40,0x85,0xe0]


@ CHECK: add	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x85,0xe2]
@ CHECK: add	r4, r4, r5              @ encoding: [0x05,0x40,0x84,0xe0]
@ CHECK: add	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0x84,0xe0]
@ CHECK: add	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x84,0xe0]
@ CHECK: add	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x84,0xe0]
@ CHECK: add	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0x84,0xe0]
@ CHECK: add	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0x84,0xe0]
@ CHECK: add	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0x86,0xe0]
@ CHECK: add	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0x86,0xe0]
@ CHECK: add	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0x86,0xe0]
@ CHECK: add	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0x86,0xe0]
@ CHECK: add	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0x84,0xe0]


@------------------------------------------------------------------------------
@ AND
@------------------------------------------------------------------------------
  and r10, r1, #0xf
  and r10, r1, r6
  and r10, r1, r6, lsl #10
  and r10, r1, r6, lsr #10
  and r10, r1, r6, lsr #10
  and r10, r1, r6, asr #10
  and r10, r1, r6, ror #10
  and r6, r7, r8, lsl r2
  and r6, r7, r8, lsr r2
  and r6, r7, r8, asr r2
  and r6, r7, r8, ror r2
  and r10, r1, r6, rrx

  @ destination register is optional
  and r1, #0xf
  and r10, r1
  and r10, r1, lsl #10
  and r10, r1, lsr #10
  and r10, r1, lsr #10
  and r10, r1, asr #10
  and r10, r1, ror #10
  and r6, r7, lsl r2
  and r6, r7, lsr r2
  and r6, r7, asr r2
  and r6, r7, ror r2
  and r10, r1, rrx

@ CHECK: and	r10, r1, #15            @ encoding: [0x0f,0xa0,0x01,0xe2]
@ CHECK: and	r10, r1, r6             @ encoding: [0x06,0xa0,0x01,0xe0]
@ CHECK: and	r10, r1, r6, lsl #10    @ encoding: [0x06,0xa5,0x01,0xe0]
@ CHECK: and	r10, r1, r6, lsr #10    @ encoding: [0x26,0xa5,0x01,0xe0]
@ CHECK: and	r10, r1, r6, lsr #10    @ encoding: [0x26,0xa5,0x01,0xe0]
@ CHECK: and	r10, r1, r6, asr #10    @ encoding: [0x46,0xa5,0x01,0xe0]
@ CHECK: and	r10, r1, r6, ror #10    @ encoding: [0x66,0xa5,0x01,0xe0]
@ CHECK: and	r6, r7, r8, lsl r2      @ encoding: [0x18,0x62,0x07,0xe0]
@ CHECK: and	r6, r7, r8, lsr r2      @ encoding: [0x38,0x62,0x07,0xe0]
@ CHECK: and	r6, r7, r8, asr r2      @ encoding: [0x58,0x62,0x07,0xe0]
@ CHECK: and	r6, r7, r8, ror r2      @ encoding: [0x78,0x62,0x07,0xe0]
@ CHECK: and	r10, r1, r6, rrx        @ encoding: [0x66,0xa0,0x01,0xe0]

@ CHECK: and	r1, r1, #15             @ encoding: [0x0f,0x10,0x01,0xe2]
@ CHECK: and	r10, r10, r1            @ encoding: [0x01,0xa0,0x0a,0xe0]
@ CHECK: and	r10, r10, r1, lsl #10   @ encoding: [0x01,0xa5,0x0a,0xe0]
@ CHECK: and	r10, r10, r1, lsr #10   @ encoding: [0x21,0xa5,0x0a,0xe0]
@ CHECK: and	r10, r10, r1, lsr #10   @ encoding: [0x21,0xa5,0x0a,0xe0]
@ CHECK: and	r10, r10, r1, asr #10   @ encoding: [0x41,0xa5,0x0a,0xe0]
@ CHECK: and	r10, r10, r1, ror #10   @ encoding: [0x61,0xa5,0x0a,0xe0]
@ CHECK: and	r6, r6, r7, lsl r2      @ encoding: [0x17,0x62,0x06,0xe0]
@ CHECK: and	r6, r6, r7, lsr r2      @ encoding: [0x37,0x62,0x06,0xe0]
@ CHECK: and	r6, r6, r7, asr r2      @ encoding: [0x57,0x62,0x06,0xe0]
@ CHECK: and	r6, r6, r7, ror r2      @ encoding: [0x77,0x62,0x06,0xe0]
@ CHECK: and	r10, r10, r1, rrx       @ encoding: [0x61,0xa0,0x0a,0xe0]

