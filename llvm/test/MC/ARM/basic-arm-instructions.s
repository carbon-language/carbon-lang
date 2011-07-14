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
@ FIXME: ADR
@------------------------------------------------------------------------------

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

@------------------------------------------------------------------------------
@ FIXME: ASR
@------------------------------------------------------------------------------
@------------------------------------------------------------------------------
@ FIXME: B
@------------------------------------------------------------------------------
@------------------------------------------------------------------------------
@ FIXME: BFC
@------------------------------------------------------------------------------
@------------------------------------------------------------------------------
@ FIXME: BFI
@------------------------------------------------------------------------------

@------------------------------------------------------------------------------
@ BIC
@------------------------------------------------------------------------------
  bic r10, r1, #0xf
  bic r10, r1, r6
  bic r10, r1, r6, lsl #10
  bic r10, r1, r6, lsr #10
  bic r10, r1, r6, lsr #10
  bic r10, r1, r6, asr #10
  bic r10, r1, r6, ror #10
  bic r6, r7, r8, lsl r2
  bic r6, r7, r8, lsr r2
  bic r6, r7, r8, asr r2
  bic r6, r7, r8, ror r2
  bic r10, r1, r6, rrx

  @ destination register is optional
  bic r1, #0xf
  bic r10, r1
  bic r10, r1, lsl #10
  bic r10, r1, lsr #10
  bic r10, r1, lsr #10
  bic r10, r1, asr #10
  bic r10, r1, ror #10
  bic r6, r7, lsl r2
  bic r6, r7, lsr r2
  bic r6, r7, asr r2
  bic r6, r7, ror r2
  bic r10, r1, rrx

@ CHECK: bic	r10, r1, #15            @ encoding: [0x0f,0xa0,0xc1,0xe3]
@ CHECK: bic	r10, r1, r6             @ encoding: [0x06,0xa0,0xc1,0xe1]
@ CHECK: bic	r10, r1, r6, lsl #10    @ encoding: [0x06,0xa5,0xc1,0xe1]
@ CHECK: bic	r10, r1, r6, lsr #10    @ encoding: [0x26,0xa5,0xc1,0xe1]
@ CHECK: bic	r10, r1, r6, lsr #10    @ encoding: [0x26,0xa5,0xc1,0xe1]
@ CHECK: bic	r10, r1, r6, asr #10    @ encoding: [0x46,0xa5,0xc1,0xe1]
@ CHECK: bic	r10, r1, r6, ror #10    @ encoding: [0x66,0xa5,0xc1,0xe1]
@ CHECK: bic	r6, r7, r8, lsl r2      @ encoding: [0x18,0x62,0xc7,0xe1]
@ CHECK: bic	r6, r7, r8, lsr r2      @ encoding: [0x38,0x62,0xc7,0xe1]
@ CHECK: bic	r6, r7, r8, asr r2      @ encoding: [0x58,0x62,0xc7,0xe1]
@ CHECK: bic	r6, r7, r8, ror r2      @ encoding: [0x78,0x62,0xc7,0xe1]
@ CHECK: bic	r10, r1, r6, rrx        @ encoding: [0x66,0xa0,0xc1,0xe1]


@ CHECK: bic	r1, r1, #15             @ encoding: [0x0f,0x10,0xc1,0xe3]
@ CHECK: bic	r10, r10, r1            @ encoding: [0x01,0xa0,0xca,0xe1]
@ CHECK: bic	r10, r10, r1, lsl #10   @ encoding: [0x01,0xa5,0xca,0xe1]
@ CHECK: bic	r10, r10, r1, lsr #10   @ encoding: [0x21,0xa5,0xca,0xe1]
@ CHECK: bic	r10, r10, r1, lsr #10   @ encoding: [0x21,0xa5,0xca,0xe1]
@ CHECK: bic	r10, r10, r1, asr #10   @ encoding: [0x41,0xa5,0xca,0xe1]
@ CHECK: bic	r10, r10, r1, ror #10   @ encoding: [0x61,0xa5,0xca,0xe1]
@ CHECK: bic	r6, r6, r7, lsl r2      @ encoding: [0x17,0x62,0xc6,0xe1]
@ CHECK: bic	r6, r6, r7, lsr r2      @ encoding: [0x37,0x62,0xc6,0xe1]
@ CHECK: bic	r6, r6, r7, asr r2      @ encoding: [0x57,0x62,0xc6,0xe1]
@ CHECK: bic	r6, r6, r7, ror r2      @ encoding: [0x77,0x62,0xc6,0xe1]
@ CHECK: bic	r10, r10, r1, rrx       @ encoding: [0x61,0xa0,0xca,0xe1]

@------------------------------------------------------------------------------
@ BKPT
@------------------------------------------------------------------------------
  bkpt #10
  bkpt #65535

@ CHECK: bkpt  #10                      @ encoding: [0x7a,0x00,0x20,0xe1]
@ CHECK: bkpt  #65535                   @ encoding: [0x7f,0xff,0x2f,0xe1]

@------------------------------------------------------------------------------
@ BL/BLX (immediate)
@------------------------------------------------------------------------------

  bl _bar
  @ FIXME: blx _bar

@ CHECK: bl  _bar @ encoding: [A,A,A,0xeb]
@ CHECK:   @   fixup A - offset: 0, value: _bar, kind: fixup_arm_uncondbranch

@------------------------------------------------------------------------------
@ BLX (register)
@------------------------------------------------------------------------------
  blx r2
  blxne r2

@ CHECK: blx r2                         @ encoding: [0x32,0xff,0x2f,0xe1]
@ CHECK: blxne r2                       @ encoding: [0x32,0xff,0x2f,0x11]

@------------------------------------------------------------------------------
@ BX
@------------------------------------------------------------------------------

  bx r2
  bxne r2

@ CHECK: bx	r2                      @ encoding: [0x12,0xff,0x2f,0xe1]
@ CHECK: bxne	r2                      @ encoding: [0x12,0xff,0x2f,0x11]

@------------------------------------------------------------------------------
@ BXJ
@------------------------------------------------------------------------------

  bxj r2
  bxjne r2

@ CHECK: bxj	r2                      @ encoding: [0x22,0xff,0x2f,0xe1]
@ CHECK: bxjne	r2                      @ encoding: [0x22,0xff,0x2f,0x11]

@------------------------------------------------------------------------------
@ FIXME: CBNZ/CBZ
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ CDP/CDP2
@------------------------------------------------------------------------------
  cdp  p7, #1, c1, c1, c1, #4
  cdp2  p7, #1, c1, c1, c1, #4

@ CHECK: cdp  p7, #1, c1, c1, c1, #4     @ encoding: [0x81,0x17,0x11,0xee]
@ CHECK: cdp2  p7, #1, c1, c1, c1, #4    @ encoding: [0x81,0x17,0x11,0xfe]


@------------------------------------------------------------------------------
@ CLREX
@------------------------------------------------------------------------------
  clrex

@ CHECK: clrex                           @ encoding: [0x1f,0xf0,0x7f,0xf5]


@------------------------------------------------------------------------------
@ CLZ
@------------------------------------------------------------------------------
  clz r1, r2
  clzeq r1, r2

@ CHECK: clz r1, r2                      @ encoding: [0x12,0x1f,0x6f,0xe1]
@ CHECK: clzeq r1, r2                    @ encoding: [0x12,0x1f,0x6f,0x01]

@------------------------------------------------------------------------------
@ CMN
@------------------------------------------------------------------------------
  cmn r1, #0xf
  cmn r1, r6
  cmn r1, r6, lsl #10
  cmn r1, r6, lsr #10
  cmn sp, r6, lsr #10
  cmn r1, r6, asr #10
  cmn r1, r6, ror #10
  cmn r7, r8, lsl r2
  cmn sp, r8, lsr r2
  cmn r7, r8, asr r2
  cmn r7, r8, ror r2
  cmn r1, r6, rrx

@ CHECK: cmn	r1, #15                 @ encoding: [0x0f,0x00,0x71,0xe3]
@ CHECK: cmn	r1, r6                  @ encoding: [0x06,0x00,0x71,0xe1]
@ CHECK: cmn	r1, r6, lsl #10         @ encoding: [0x06,0x05,0x71,0xe1]
@ CHECK: cmn	r1, r6, lsr #10         @ encoding: [0x26,0x05,0x71,0xe1]
@ CHECK: cmn	sp, r6, lsr #10         @ encoding: [0x26,0x05,0x7d,0xe1]
@ CHECK: cmn	r1, r6, asr #10         @ encoding: [0x46,0x05,0x71,0xe1]
@ CHECK: cmn	r1, r6, ror #10         @ encoding: [0x66,0x05,0x71,0xe1]
@ CHECK: cmn	r7, r8, lsl r2          @ encoding: [0x18,0x02,0x77,0xe1]
@ CHECK: cmn	sp, r8, lsr r2          @ encoding: [0x38,0x02,0x7d,0xe1]
@ CHECK: cmn	r7, r8, asr r2          @ encoding: [0x58,0x02,0x77,0xe1]
@ CHECK: cmn	r7, r8, ror r2          @ encoding: [0x78,0x02,0x77,0xe1]
@ CHECK: cmn	r1, r6, rrx             @ encoding: [0x66,0x00,0x71,0xe1]

@------------------------------------------------------------------------------
@ CMP
@------------------------------------------------------------------------------
  cmp r1, #0xf
  cmp r1, r6
  cmp r1, r6, lsl #10
  cmp r1, r6, lsr #10
  cmp sp, r6, lsr #10
  cmp r1, r6, asr #10
  cmp r1, r6, ror #10
  cmp r7, r8, lsl r2
  cmp sp, r8, lsr r2
  cmp r7, r8, asr r2
  cmp r7, r8, ror r2
  cmp r1, r6, rrx

@ CHECK: cmp	r1, #15                 @ encoding: [0x0f,0x00,0x51,0xe3]
@ CHECK: cmp	r1, r6                  @ encoding: [0x06,0x00,0x51,0xe1]
@ CHECK: cmp	r1, r6, lsl #10         @ encoding: [0x06,0x05,0x51,0xe1]
@ CHECK: cmp	r1, r6, lsr #10         @ encoding: [0x26,0x05,0x51,0xe1]
@ CHECK: cmp	sp, r6, lsr #10         @ encoding: [0x26,0x05,0x5d,0xe1]
@ CHECK: cmp	r1, r6, asr #10         @ encoding: [0x46,0x05,0x51,0xe1]
@ CHECK: cmp	r1, r6, ror #10         @ encoding: [0x66,0x05,0x51,0xe1]
@ CHECK: cmp	r7, r8, lsl r2          @ encoding: [0x18,0x02,0x57,0xe1]
@ CHECK: cmp	sp, r8, lsr r2          @ encoding: [0x38,0x02,0x5d,0xe1]
@ CHECK: cmp	r7, r8, asr r2          @ encoding: [0x58,0x02,0x57,0xe1]
@ CHECK: cmp	r7, r8, ror r2          @ encoding: [0x78,0x02,0x57,0xe1]
@ CHECK: cmp	r1, r6, rrx             @ encoding: [0x66,0x00,0x51,0xe1]

@------------------------------------------------------------------------------
@ DBG
@------------------------------------------------------------------------------
  dbg #0
  dbg #5
  dbg #15

@ CHECK: dbg #0                         @ encoding: [0xf0,0xf0,0x20,0xe3]
@ CHECK: dbg #5                         @ encoding: [0xf5,0xf0,0x20,0xe3]
@ CHECK: dbg #15                        @ encoding: [0xff,0xf0,0x20,0xe3]


@------------------------------------------------------------------------------
@ DMB
@------------------------------------------------------------------------------
  dmb sy
  dmb st
  dmb sh
  dmb ish
  dmb shst
  dmb ishst
  dmb un
  dmb nsh
  dmb unst
  dmb nshst
  dmb osh
  dmb oshst
  dmb

@ CHECK: dmb	sy                      @ encoding: [0x5f,0xf0,0x7f,0xf5]
@ CHECK: dmb	st                      @ encoding: [0x5e,0xf0,0x7f,0xf5]
@ CHECK: dmb	ish                     @ encoding: [0x5b,0xf0,0x7f,0xf5]
@ CHECK: dmb	ish                     @ encoding: [0x5b,0xf0,0x7f,0xf5]
@ CHECK: dmb	ishst                   @ encoding: [0x5a,0xf0,0x7f,0xf5]
@ CHECK: dmb	ishst                   @ encoding: [0x5a,0xf0,0x7f,0xf5]
@ CHECK: dmb	nsh                     @ encoding: [0x57,0xf0,0x7f,0xf5]
@ CHECK: dmb	nsh                     @ encoding: [0x57,0xf0,0x7f,0xf5]
@ CHECK: dmb	nshst                   @ encoding: [0x56,0xf0,0x7f,0xf5]
@ CHECK: dmb	nshst                   @ encoding: [0x56,0xf0,0x7f,0xf5]
@ CHECK: dmb	osh                     @ encoding: [0x53,0xf0,0x7f,0xf5]
@ CHECK: dmb	oshst                   @ encoding: [0x52,0xf0,0x7f,0xf5]
@ CHECK: dmb	sy                      @ encoding: [0x5f,0xf0,0x7f,0xf5]

@------------------------------------------------------------------------------
@ DSB
@------------------------------------------------------------------------------
  dsb sy
  dsb st
  dsb sh
  dsb ish
  dsb shst
  dsb ishst
  dsb un
  dsb nsh
  dsb unst
  dsb nshst
  dsb osh
  dsb oshst
  dsb

@ CHECK: dsb	sy                      @ encoding: [0x4f,0xf0,0x7f,0xf5]
@ CHECK: dsb	st                      @ encoding: [0x4e,0xf0,0x7f,0xf5]
@ CHECK: dsb	ish                     @ encoding: [0x4b,0xf0,0x7f,0xf5]
@ CHECK: dsb	ish                     @ encoding: [0x4b,0xf0,0x7f,0xf5]
@ CHECK: dsb	ishst                   @ encoding: [0x4a,0xf0,0x7f,0xf5]
@ CHECK: dsb	ishst                   @ encoding: [0x4a,0xf0,0x7f,0xf5]
@ CHECK: dsb	nsh                     @ encoding: [0x47,0xf0,0x7f,0xf5]
@ CHECK: dsb	nsh                     @ encoding: [0x47,0xf0,0x7f,0xf5]
@ CHECK: dsb	nshst                   @ encoding: [0x46,0xf0,0x7f,0xf5]
@ CHECK: dsb	nshst                   @ encoding: [0x46,0xf0,0x7f,0xf5]
@ CHECK: dsb	osh                     @ encoding: [0x43,0xf0,0x7f,0xf5]
@ CHECK: dsb	oshst                   @ encoding: [0x42,0xf0,0x7f,0xf5]
@ CHECK: dsb	sy                      @ encoding: [0x4f,0xf0,0x7f,0xf5]

@------------------------------------------------------------------------------
@ EOR
@------------------------------------------------------------------------------
  eor r4, r5, #0xf000
  eor r4, r5, r6
  eor r4, r5, r6, lsl #5
  eor r4, r5, r6, lsr #5
  eor r4, r5, r6, lsr #5
  eor r4, r5, r6, asr #5
  eor r4, r5, r6, ror #5
  eor r6, r7, r8, lsl r9
  eor r6, r7, r8, lsr r9
  eor r6, r7, r8, asr r9
  eor r6, r7, r8, ror r9
  eor r4, r5, r6, rrx

  @ destination register is optional
  eor r5, #0xf000
  eor r4, r5
  eor r4, r5, lsl #5
  eor r4, r5, lsr #5
  eor r4, r5, lsr #5
  eor r4, r5, asr #5
  eor r4, r5, ror #5
  eor r6, r7, lsl r9
  eor r6, r7, lsr r9
  eor r6, r7, asr r9
  eor r6, r7, ror r9
  eor r4, r5, rrx

@ CHECK: eor	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x25,0xe2]
@ CHECK: eor	r4, r5, r6              @ encoding: [0x06,0x40,0x25,0xe0]
@ CHECK: eor	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0x25,0xe0]
@ CHECK: eor	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x25,0xe0]
@ CHECK: eor	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x25,0xe0]
@ CHECK: eor	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0x25,0xe0]
@ CHECK: eor	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0x25,0xe0]
@ CHECK: eor	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0x27,0xe0]
@ CHECK: eor	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0x27,0xe0]
@ CHECK: eor	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0x27,0xe0]
@ CHECK: eor	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0x27,0xe0]
@ CHECK: eor	r4, r5, r6, rrx         @ encoding: [0x66,0x40,0x25,0xe0]


@ CHECK: eor	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x25,0xe2]
@ CHECK: eor	r4, r4, r5              @ encoding: [0x05,0x40,0x24,0xe0]
@ CHECK: eor	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0x24,0xe0]
@ CHECK: eor	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x24,0xe0]
@ CHECK: eor	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x24,0xe0]
@ CHECK: eor	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0x24,0xe0]
@ CHECK: eor	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0x24,0xe0]
@ CHECK: eor	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0x26,0xe0]
@ CHECK: eor	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0x26,0xe0]
@ CHECK: eor	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0x26,0xe0]
@ CHECK: eor	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0x26,0xe0]
@ CHECK: eor	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0x24,0xe0]


@------------------------------------------------------------------------------
@ ISB
@------------------------------------------------------------------------------
        isb sy
        isb

@ CHECK: isb sy                         @ encoding: [0x6f,0xf0,0x7f,0xf5]
@ CHECK: isb sy                         @ encoding: [0x6f,0xf0,0x7f,0xf5]



@------------------------------------------------------------------------------
@ LDM*
@------------------------------------------------------------------------------
        ldm       r2, {r1,r3-r6,sp}
        ldmia     r2, {r1,r3-r6,sp}
        ldmib     r2, {r1,r3-r6,sp}
        ldmda     r2, {r1,r3-r6,sp}
        ldmdb     r2, {r1,r3-r6,sp}
        ldmfd     r2, {r1,r3-r6,sp}

        @ with update
        ldm       r2!, {r1,r3-r6,sp}
        ldmib     r2!, {r1,r3-r6,sp}
        ldmda     r2!, {r1,r3-r6,sp}
        ldmdb     r2!, {r1,r3-r6,sp}

@ CHECK: ldm   r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x92,0xe8]
@ CHECK: ldm   r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x92,0xe8]
@ CHECK: ldmib r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x92,0xe9]
@ CHECK: ldmda r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x12,0xe8]
@ CHECK: ldmdb r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x12,0xe9]
@ CHECK: ldm   r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x92,0xe8]

@ CHECK: ldm   r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xb2,0xe8]
@ CHECK: ldmib r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xb2,0xe9]
@ CHECK: ldmda r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x32,0xe8]
@ CHECK: ldmdb r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x32,0xe9]

@------------------------------------------------------------------------------
@ FIXME: LDR*
@------------------------------------------------------------------------------
@------------------------------------------------------------------------------
@ FIXME: LSL
@------------------------------------------------------------------------------
@------------------------------------------------------------------------------
@ FIXME: LSR
@------------------------------------------------------------------------------

@------------------------------------------------------------------------------
@ MCR/MCR2
@------------------------------------------------------------------------------
        mcr  p7, #1, r5, c1, c1, #4
        mcr2  p7, #1, r5, c1, c1, #4

@ CHECK: mcr  p7, #1, r5, c1, c1, #4 @ encoding: [0x91,0x57,0x21,0xee]
@ CHECK: mcr2  p7, #1, r5, c1, c1, #4 @ encoding: [0x91,0x57,0x21,0xfe]

@------------------------------------------------------------------------------
@ STM*
@------------------------------------------------------------------------------
        stm       r2, {r1,r3-r6,sp}
        stmia     r2, {r1,r3-r6,sp}
        stmib     r2, {r1,r3-r6,sp}
        stmda     r2, {r1,r3-r6,sp}
        stmdb     r2, {r1,r3-r6,sp}
        stmfd     r2, {r1,r3-r6,sp}

        @ with update
        stmia     r2!, {r1,r3-r6,sp}
        stmib     r2!, {r1,r3-r6,sp}
        stmda     r2!, {r1,r3-r6,sp}
        stmdb     r2!, {r1,r3-r6,sp}
@ CHECK: stm   r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x82,0xe8]
@ CHECK: stm   r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x82,0xe8]
@ CHECK: stmib r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x82,0xe9]
@ CHECK: stmda r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x02,0xe8]
@ CHECK: stmdb r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x02,0xe9]
@ CHECK: stmdb r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x02,0xe9]

@ CHECK: stm   r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xa2,0xe8]
@ CHECK: stmib r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xa2,0xe9]
@ CHECK: stmda r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x22,0xe8]
@ CHECK: stmdb r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x22,0xe9]
