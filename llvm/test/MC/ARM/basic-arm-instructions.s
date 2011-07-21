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
@ MCRR/MCRR2
@------------------------------------------------------------------------------
        mcrr  p7, #15, r5, r4, c1
        mcrr2  p7, #15, r5, r4, c1

@ CHECK: mcrr  p7, #15, r5, r4, c1 @ encoding: [0xf1,0x57,0x44,0xec]
@ CHECK: mcrr2  p7, #15, r5, r4, c1 @ encoding: [0xf1,0x57,0x44,0xfc]


@------------------------------------------------------------------------------
@ MLA
@------------------------------------------------------------------------------
        mla  r1,r2,r3,r4
        mlas r1,r2,r3,r4
        mlane  r1,r2,r3,r4
        mlasne r1,r2,r3,r4

@ CHECK: mla 	r1, r2, r3, r4 @ encoding: [0x92,0x43,0x21,0xe0]
@ CHECK: mlas	r1, r2, r3, r4 @ encoding: [0x92,0x43,0x31,0xe0]
@ CHECK: mlane 	r1, r2, r3, r4 @ encoding: [0x92,0x43,0x21,0x10]
@ CHECK: mlasne	r1, r2, r3, r4 @ encoding: [0x92,0x43,0x31,0x10]

@------------------------------------------------------------------------------
@ MLS
@------------------------------------------------------------------------------
        mls  r2,r5,r6,r3
        mlsne  r2,r5,r6,r3

@ CHECK: mls	r2, r5, r6, r3          @ encoding: [0x95,0x36,0x62,0xe0]
@ CHECK: mlsne	r2, r5, r6, r3          @ encoding: [0x95,0x36,0x62,0x10]

@------------------------------------------------------------------------------
@ MOV (immediate)
@------------------------------------------------------------------------------
    mov r3, #7
    mov r4, #0xff0
    mov r5, #0xff0000
    mov r6, #0xffff
    movw r9, #0xffff
    movs r3, #7
    moveq r4, #0xff0
    movseq r5, #0xff0000

@ CHECK: mov	r3, #7                  @ encoding: [0x07,0x30,0xa0,0xe3]
@ CHECK: mov	r4, #4080               @ encoding: [0xff,0x4e,0xa0,0xe3]
@ CHECK: mov	r5, #16711680           @ encoding: [0xff,0x58,0xa0,0xe3]
@ CHECK: movw	r6, #65535              @ encoding: [0xff,0x6f,0x0f,0xe3]
@ CHECK: movw	r9, #65535              @ encoding: [0xff,0x9f,0x0f,0xe3]
@ CHECK: movs	r3, #7                  @ encoding: [0x07,0x30,0xb0,0xe3]
@ CHECK: moveq	r4, #4080               @ encoding: [0xff,0x4e,0xa0,0x03]
@ CHECK: movseq	r5, #16711680           @ encoding: [0xff,0x58,0xb0,0x03]

@------------------------------------------------------------------------------
@ MOV (register)
@------------------------------------------------------------------------------
        mov r2, r3
        movs r2, r3
        moveq r2, r3
        movseq r2, r3

@ CHECK: mov	r2, r3                  @ encoding: [0x03,0x20,0xa0,0xe1]
@ CHECK: movs	r2, r3                  @ encoding: [0x03,0x20,0xb0,0xe1]
@ CHECK: moveq	r2, r3                  @ encoding: [0x03,0x20,0xa0,0x01]
@ CHECK: movseq	r2, r3                  @ encoding: [0x03,0x20,0xb0,0x01]

@------------------------------------------------------------------------------
@ MOVT
@------------------------------------------------------------------------------
    movt r3, #7
    movt r6, #0xffff
    movteq r4, #0xff0

@ CHECK: movt	r3, #7                  @ encoding: [0x07,0x30,0x40,0xe3]
@ CHECK: movt	r6, #65535              @ encoding: [0xff,0x6f,0x4f,0xe3]
@ CHECK: movteq	r4, #4080               @ encoding: [0xf0,0x4f,0x40,0x03]


@------------------------------------------------------------------------------
@ MRC/MRC2
@------------------------------------------------------------------------------
        mrc  p14, #0, r1, c1, c2, #4
        mrc2  p14, #0, r1, c1, c2, #4

@ CHECK: mrc  p14, #0, r1, c1, c2, #4   @ encoding: [0x92,0x1e,0x11,0xee]
@ CHECK: mrc2  p14, #0, r1, c1, c2, #4  @ encoding: [0x92,0x1e,0x11,0xfe]

@------------------------------------------------------------------------------
@ MRRC/MRRC2
@------------------------------------------------------------------------------
        mrrc  p7, #1, r5, r4, c1
        mrrc2  p7, #1, r5, r4, c1

@ CHECK: mrrc  p7, #1, r5, r4, c1       @ encoding: [0x11,0x57,0x54,0xec]
@ CHECK: mrrc2  p7, #1, r5, r4, c1      @ encoding: [0x11,0x57,0x54,0xfc]


@------------------------------------------------------------------------------
@ MRS
@------------------------------------------------------------------------------
        mrs  r8, apsr
        mrs  r8, cpsr
        mrs  r8, spsr
@ CHECK: mrs  r8, apsr @ encoding: [0x00,0x80,0x0f,0xe1]
@ CHECK: mrs  r8, apsr @ encoding: [0x00,0x80,0x0f,0xe1]
@ CHECK: mrs  r8, spsr @ encoding: [0x00,0x80,0x4f,0xe1]



@------------------------------------------------------------------------------
@ MSR
@------------------------------------------------------------------------------

        msr  apsr, #5
        msr  apsr_g, #5
        msr  apsr_nzcvq, #5
        msr  APSR_nzcvq, #5
        msr  apsr_nzcvqg, #5
        msr  cpsr_fc, #5
        msr  cpsr_c, #5
        msr  cpsr_x, #5
        msr  cpsr_fc, #5
        msr  cpsr_all, #5
        msr  cpsr_fsx, #5
        msr  spsr_fc, #5
        msr  SPSR_fsxc, #5
        msr  cpsr_fsxc, #5

@ CHECK: msr	CPSR_fc, #5             @ encoding: [0x05,0xf0,0x29,0xe3]
@ CHECK: msr	APSR_g, #5              @ encoding: [0x05,0xf0,0x24,0xe3]
@ CHECK: msr	APSR_nzcvq, #5          @ encoding: [0x05,0xf0,0x28,0xe3]
@ CHECK: msr	APSR_nzcvq, #5          @ encoding: [0x05,0xf0,0x28,0xe3]
@ CHECK: msr	APSR_nzcvqg, #5         @ encoding: [0x05,0xf0,0x2c,0xe3]
@ CHECK: msr	CPSR_fc, #5             @ encoding: [0x05,0xf0,0x29,0xe3]
@ CHECK: msr	CPSR_c, #5              @ encoding: [0x05,0xf0,0x21,0xe3]
@ CHECK: msr	CPSR_x, #5              @ encoding: [0x05,0xf0,0x22,0xe3]
@ CHECK: msr	CPSR_fc, #5             @ encoding: [0x05,0xf0,0x29,0xe3]
@ CHECK: msr	CPSR_fc, #5             @ encoding: [0x05,0xf0,0x29,0xe3]
@ CHECK: msr	CPSR_fsx, #5            @ encoding: [0x05,0xf0,0x2e,0xe3]
@ CHECK: msr	SPSR_fc, #5             @ encoding: [0x05,0xf0,0x69,0xe3]
@ CHECK: msr	SPSR_fsxc, #5           @ encoding: [0x05,0xf0,0x6f,0xe3]
@ CHECK: msr	CPSR_fsxc, #5           @ encoding: [0x05,0xf0,0x2f,0xe3]

        msr  apsr, r0
        msr  apsr_g, r0
        msr  apsr_nzcvq, r0
        msr  APSR_nzcvq, r0
        msr  apsr_nzcvqg, r0
        msr  cpsr_fc, r0
        msr  cpsr_c, r0
        msr  cpsr_x, r0
        msr  cpsr_fc, r0
        msr  cpsr_all, r0
        msr  cpsr_fsx, r0
        msr  spsr_fc, r0
        msr  SPSR_fsxc, r0
        msr  cpsr_fsxc, r0

@ CHECK: msr  CPSR_fc, r0 @ encoding: [0x00,0xf0,0x29,0xe1]
@ CHECK: msr  APSR_g, r0 @ encoding: [0x00,0xf0,0x24,0xe1]
@ CHECK: msr  APSR_nzcvq, r0 @ encoding: [0x00,0xf0,0x28,0xe1]
@ CHECK: msr  APSR_nzcvq, r0 @ encoding: [0x00,0xf0,0x28,0xe1]
@ CHECK: msr  APSR_nzcvqg, r0 @ encoding: [0x00,0xf0,0x2c,0xe1]
@ CHECK: msr  CPSR_fc, r0 @ encoding: [0x00,0xf0,0x29,0xe1]
@ CHECK: msr  CPSR_c, r0 @ encoding: [0x00,0xf0,0x21,0xe1]
@ CHECK: msr  CPSR_x, r0 @ encoding: [0x00,0xf0,0x22,0xe1]
@ CHECK: msr  CPSR_fc, r0 @ encoding: [0x00,0xf0,0x29,0xe1]
@ CHECK: msr  CPSR_fc, r0 @ encoding: [0x00,0xf0,0x29,0xe1]
@ CHECK: msr  CPSR_fsx, r0 @ encoding: [0x00,0xf0,0x2e,0xe1]
@ CHECK: msr  SPSR_fc, r0 @ encoding: [0x00,0xf0,0x69,0xe1]
@ CHECK: msr  SPSR_fsxc, r0 @ encoding: [0x00,0xf0,0x6f,0xe1]
@ CHECK: msr  CPSR_fsxc, r0 @ encoding: [0x00,0xf0,0x2f,0xe1]

@------------------------------------------------------------------------------
@ MUL
@------------------------------------------------------------------------------

  mul r5, r6, r7
  muls r5, r6, r7
  mulgt r5, r6, r7
  mulsle r5, r6, r7

@ CHECK: mul	r5, r6, r7              @ encoding: [0x96,0x07,0x05,0xe0]
@ CHECK: muls	r5, r6, r7              @ encoding: [0x96,0x07,0x15,0xe0]
@ CHECK: mulgt	r5, r6, r7              @ encoding: [0x96,0x07,0x05,0xc0]
@ CHECK: mulsle	r5, r6, r7              @ encoding: [0x96,0x07,0x15,0xd0]


@------------------------------------------------------------------------------
@ MVN (immediate)
@------------------------------------------------------------------------------
    mvn r3, #7
    mvn r4, #0xff0
    mvn r5, #0xff0000
    mvns r3, #7
    mvneq r4, #0xff0
    mvnseq r5, #0xff0000

@ CHECK: mvn	r3, #7                  @ encoding: [0x07,0x30,0xe0,0xe3]
@ CHECK: mvn	r4, #4080               @ encoding: [0xff,0x4e,0xe0,0xe3]
@ CHECK: mvn	r5, #16711680           @ encoding: [0xff,0x58,0xe0,0xe3]
@ CHECK: mvns	r3, #7                  @ encoding: [0x07,0x30,0xf0,0xe3]
@ CHECK: mvneq	r4, #4080               @ encoding: [0xff,0x4e,0xe0,0x03]
@ CHECK: mvnseq	r5, #16711680           @ encoding: [0xff,0x58,0xf0,0x03]


@------------------------------------------------------------------------------
@ MVN (register)
@------------------------------------------------------------------------------
        mvn r2, r3
        mvns r2, r3
        mvn r5, r6, lsl #19
        mvn r5, r6, lsr #9
        mvn r5, r6, asr #4
        mvn r5, r6, ror #6
        mvn r5, r6, rrx
        mvneq r2, r3
        mvnseq r2, r3, lsl #10

@ CHECK: mvn	r2, r3                  @ encoding: [0x03,0x20,0xe0,0xe1]
@ CHECK: mvns	r2, r3                  @ encoding: [0x03,0x20,0xf0,0xe1]
@ CHECK: mvn	r5, r6, lsl #19         @ encoding: [0x86,0x59,0xe0,0xe1]
@ CHECK: mvn	r5, r6, lsr #9          @ encoding: [0xa6,0x54,0xe0,0xe1]
@ CHECK: mvn	r5, r6, asr #4          @ encoding: [0x46,0x52,0xe0,0xe1]
@ CHECK: mvn	r5, r6, ror #6          @ encoding: [0x66,0x53,0xe0,0xe1]
@ CHECK: mvn	r5, r6, rrx             @ encoding: [0x66,0x50,0xe0,0xe1]
@ CHECK: mvneq	r2, r3                  @ encoding: [0x03,0x20,0xe0,0x01]
@ CHECK: mvnseq	r2, r3, lsl #10         @ encoding: [0x03,0x25,0xf0,0x01]


@------------------------------------------------------------------------------
@ MVN (shifted register)
@------------------------------------------------------------------------------
        mvn r5, r6, lsl r7
        mvns r5, r6, lsr r7
        mvngt r5, r6, asr r7
        mvnslt r5, r6, ror r7

@ CHECK: mvn	r5, r6, lsl r7          @ encoding: [0x16,0x57,0xe0,0xe1]
@ CHECK: mvns	r5, r6, lsr r7          @ encoding: [0x36,0x57,0xf0,0xe1]
@ CHECK: mvngt	r5, r6, asr r7          @ encoding: [0x56,0x57,0xe0,0xc1]
@ CHECK: mvnslt	r5, r6, ror r7          @ encoding: [0x76,0x57,0xf0,0xb1]

@------------------------------------------------------------------------------
@ NOP
@------------------------------------------------------------------------------
        nop
        nopgt

@ CHECK: nop @ encoding: [0x00,0xf0,0x20,0xe3]
@ CHECK: nopgt @ encoding: [0x00,0xf0,0x20,0xc3]


@------------------------------------------------------------------------------
@ ORR
@------------------------------------------------------------------------------
        orr r4, r5, #0xf000
        orr r4, r5, r6
        orr r4, r5, r6, lsl #5
        orr r4, r5, r6, lsr #5
        orr r4, r5, r6, lsr #5
        orr r4, r5, r6, asr #5
        orr r4, r5, r6, ror #5
        orr r6, r7, r8, lsl r9
        orr r6, r7, r8, lsr r9
        orr r6, r7, r8, asr r9
        orr r6, r7, r8, ror r9
        orr r4, r5, r6, rrx

        @ destination register is optional
        orr r5, #0xf000
        orr r4, r5
        orr r4, r5, lsl #5
        orr r4, r5, lsr #5
        orr r4, r5, lsr #5
        orr r4, r5, asr #5
        orr r4, r5, ror #5
        orr r6, r7, lsl r9
        orr r6, r7, lsr r9
        orr r6, r7, asr r9
        orr r6, r7, ror r9
        orr r4, r5, rrx

@ CHECK: orr	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x85,0xe3]
@ CHECK: orr	r4, r5, r6              @ encoding: [0x06,0x40,0x85,0xe1]
@ CHECK: orr	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0x85,0xe1]
@ CHECK: orr	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x85,0xe1]
@ CHECK: orr	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x85,0xe1]
@ CHECK: orr	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0x85,0xe1]
@ CHECK: orr	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0x85,0xe1]
@ CHECK: orr	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0x87,0xe1]
@ CHECK: orr	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0x87,0xe1]
@ CHECK: orr	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0x87,0xe1]
@ CHECK: orr	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0x87,0xe1]
@ CHECK: orr	r4, r5, r6, rrx         @ encoding: [0x66,0x40,0x85,0xe1]

@ CHECK: orr	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x85,0xe3]
@ CHECK: orr	r4, r4, r5              @ encoding: [0x05,0x40,0x84,0xe1]
@ CHECK: orr	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0x84,0xe1]
@ CHECK: orr	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x84,0xe1]
@ CHECK: orr	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x84,0xe1]
@ CHECK: orr	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0x84,0xe1]
@ CHECK: orr	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0x84,0xe1]
@ CHECK: orr	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0x86,0xe1]
@ CHECK: orr	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0x86,0xe1]
@ CHECK: orr	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0x86,0xe1]
@ CHECK: orr	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0x86,0xe1]
@ CHECK: orr	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0x84,0xe1]

        orrseq r4, r5, #0xf000
        orrne r4, r5, r6
        orrseq r4, r5, r6, lsl #5
        orrlo r6, r7, r8, ror r9
        orrshi r4, r5, r6, rrx
        orrcs r5, #0xf000
        orrseq r4, r5
        orrne r6, r7, asr r9
        orrslt r6, r7, ror r9
        orrsgt r4, r5, rrx

@ CHECK: orrseq	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x95,0x03]
@ CHECK: orrne	r4, r5, r6              @ encoding: [0x06,0x40,0x85,0x11]
@ CHECK: orrseq	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0x95,0x01]
@ CHECK: orrlo	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0x87,0x31]
@ CHECK: orrshi	r4, r5, r6, rrx         @ encoding: [0x66,0x40,0x95,0x81]
@ CHECK: orrhs	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x85,0x23]
@ CHECK: orrseq	r4, r4, r5              @ encoding: [0x05,0x40,0x94,0x01]
@ CHECK: orrne	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0x86,0x11]
@ CHECK: orrslt	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0x96,0xb1]
@ CHECK: orrsgt	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0x94,0xc1]

@------------------------------------------------------------------------------
@ PKH
@------------------------------------------------------------------------------
        pkhbt r2, r2, r3
        pkhbt r2, r2, r3, lsl #31
        pkhbt r2, r2, r3, lsl #0
        pkhbt r2, r2, r3, lsl #15

        pkhtb r2, r2, r3
        pkhtb r2, r2, r3, asr #31
        pkhtb r2, r2, r3, asr #15

@ CHECK: pkhbt	r2, r2, r3              @ encoding: [0x13,0x20,0x82,0xe6]
@ CHECK: pkhbt	r2, r2, r3, lsl #31     @ encoding: [0x93,0x2f,0x82,0xe6]
@ CHECK: pkhbt	r2, r2, r3              @ encoding: [0x13,0x20,0x82,0xe6]
@ CHECK: pkhbt	r2, r2, r3, lsl #15     @ encoding: [0x93,0x27,0x82,0xe6]

@ CHECK: pkhbt	r2, r2, r3              @ encoding: [0x13,0x20,0x82,0xe6]
@ CHECK: pkhtb	r2, r2, r3, asr #31     @ encoding: [0xd3,0x2f,0x82,0xe6]
@ CHECK: pkhtb	r2, r2, r3, asr #15     @ encoding: [0xd3,0x27,0x82,0xe6]

@------------------------------------------------------------------------------
@ FIXME: PLD
@------------------------------------------------------------------------------
@------------------------------------------------------------------------------
@ FIXME: PLI
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ POP
@------------------------------------------------------------------------------
        pop {r7}
        pop {r7, r8, r9, r10}

@ FIXME: pop of a single register should encode as "ldr r7, [sp], #4"
@ CHECK-FIXME: pop	{r7}                    @ encoding: [0x04,0x70,0x9d,0xe4]
@ CHECK: pop	{r7, r8, r9, r10}       @ encoding: [0x80,0x07,0xbd,0xe8]


@------------------------------------------------------------------------------
@ PUSH
@------------------------------------------------------------------------------
        push {r7}
        push {r7, r8, r9, r10}

@ FIXME: push of a single register should encode as "str r7, [sp, #-4]!"
@ CHECK-FIXME: push	{r7}                    @ encoding: [0x04,0x70,0x2d,0xe5]
@ CHECK: push	{r7, r8, r9, r10}       @ encoding: [0x80,0x07,0x2d,0xe9]


@------------------------------------------------------------------------------
@ QADD/QADD16/QADD8
@------------------------------------------------------------------------------
        qadd r1, r2, r3
        qaddne r1, r2, r3
        qadd16 r1, r2, r3
        qadd16gt r1, r2, r3
        qadd8 r1, r2, r3
        qadd8le r1, r2, r3

@ CHECK: qadd	r1, r2, r3              @ encoding: [0x52,0x10,0x03,0xe1]
@ CHECK: qaddne	r1, r2, r3              @ encoding: [0x52,0x10,0x03,0x11]
@ CHECK: qadd16	r1, r2, r3              @ encoding: [0x13,0x1f,0x22,0xe6]
@ CHECK: qadd16gt	r1, r2, r3      @ encoding: [0x13,0x1f,0x22,0xc6]
@ CHECK: qadd8	r1, r2, r3              @ encoding: [0x93,0x1f,0x22,0xe6]
@ CHECK: qadd8le r1, r2, r3             @ encoding: [0x93,0x1f,0x22,0xd6]


@------------------------------------------------------------------------------
@ QDADD/QDSUB
@------------------------------------------------------------------------------
        qdadd r6, r7, r8
        qdaddhi r6, r7, r8
        qdsub r6, r7, r8
        qdsubhi r6, r7, r8

@ CHECK: qdadd	r6, r7, r8              @ encoding: [0x57,0x60,0x48,0xe1]
@ CHECK: qdaddhi r6, r7, r8             @ encoding: [0x57,0x60,0x48,0x81]
@ CHECK: qdsub	r6, r7, r8              @ encoding: [0x57,0x60,0x68,0xe1]
@ CHECK: qdsubhi r6, r7, r8             @ encoding: [0x57,0x60,0x68,0x81]


@------------------------------------------------------------------------------
@ QSAX
@------------------------------------------------------------------------------
        qsax r9, r12, r0
        qsaxeq r9, r12, r0

@ CHECK: qsax	r9, r12, r0             @ encoding: [0x50,0x9f,0x2c,0xe6]
@ CHECK: qsaxeq	r9, r12, r0             @ encoding: [0x50,0x9f,0x2c,0x06]


@------------------------------------------------------------------------------
@ QADD/QADD16/QADD8
@------------------------------------------------------------------------------
        qsub r1, r2, r3
        qsubne r1, r2, r3
        qsub16 r1, r2, r3
        qsub16gt r1, r2, r3
        qsub8 r1, r2, r3
        qsub8le r1, r2, r3

@ CHECK: qsub	r1, r2, r3              @ encoding: [0x52,0x10,0x23,0xe1]
@ CHECK: qsubne	r1, r2, r3              @ encoding: [0x52,0x10,0x23,0x11]
@ CHECK: qsub16	r1, r2, r3              @ encoding: [0x73,0x1f,0x22,0xe6]
@ CHECK: qsub16gt	r1, r2, r3      @ encoding: [0x73,0x1f,0x22,0xc6]
@ CHECK: qsub8	r1, r2, r3              @ encoding: [0xf3,0x1f,0x22,0xe6]
@ CHECK: qsub8le r1, r2, r3             @ encoding: [0xf3,0x1f,0x22,0xd6]


@------------------------------------------------------------------------------
@ RBIT
@------------------------------------------------------------------------------
        rbit r1, r2
        rbitne r1, r2

@ CHECK: rbit	r1, r2                  @ encoding: [0x32,0x1f,0xff,0xe6]
@ CHECK: rbitne	r1, r2                  @ encoding: [0x32,0x1f,0xff,0x16]


@------------------------------------------------------------------------------
@ REV/REV16/REVSH
@------------------------------------------------------------------------------
        rev r1, r9
        revne r1, r5
        rev16 r8, r3
        rev16ne r12, r4
        revsh r4, r9
        revshne r9, r1

@ CHECK: rev	r1, r9                  @ encoding: [0x39,0x1f,0xbf,0xe6]
@ CHECK: revne	r1, r5                  @ encoding: [0x35,0x1f,0xbf,0x16]
@ CHECK: rev16	r8, r3                  @ encoding: [0xb3,0x8f,0xbf,0xe6]
@ CHECK: rev16ne r12, r4                @ encoding: [0xb4,0xcf,0xbf,0x16]
@ CHECK: revsh	r4, r9                  @ encoding: [0xb9,0x4f,0xff,0xe6]
@ CHECK: revshne r9, r1                 @ encoding: [0xb1,0x9f,0xff,0x16]


@------------------------------------------------------------------------------
@ RSB
@------------------------------------------------------------------------------
        rsb r4, r5, #0xf000
        rsb r4, r5, r6
        rsb r4, r5, r6, lsl #5
        rsblo r4, r5, r6, lsr #5
        rsb r4, r5, r6, lsr #5
        rsb r4, r5, r6, asr #5
        rsb r4, r5, r6, ror #5
        rsb r6, r7, r8, lsl r9
        rsb r6, r7, r8, lsr r9
        rsb r6, r7, r8, asr r9
        rsble r6, r7, r8, ror r9
        rsb r4, r5, r6, rrx

        @ destination register is optional
        rsb r5, #0xf000
        rsb r4, r5
        rsb r4, r5, lsl #5
        rsb r4, r5, lsr #5
        rsbne r4, r5, lsr #5
        rsb r4, r5, asr #5
        rsb r4, r5, ror #5
        rsbgt r6, r7, lsl r9
        rsb r6, r7, lsr r9
        rsb r6, r7, asr r9
        rsb r6, r7, ror r9
        rsb r4, r5, rrx

@ CHECK: rsb	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x65,0xe2]
@ CHECK: rsb	r4, r5, r6              @ encoding: [0x06,0x40,0x65,0xe0]
@ CHECK: rsb	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0x65,0xe0]
@ CHECK: rsblo	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x65,0x30]
@ CHECK: rsb	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x65,0xe0]
@ CHECK: rsb	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0x65,0xe0]
@ CHECK: rsb	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0x65,0xe0]
@ CHECK: rsb	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0x67,0xe0]
@ CHECK: rsb	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0x67,0xe0]
@ CHECK: rsb	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0x67,0xe0]
@ CHECK: rsble	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0x67,0xd0]
@ CHECK: rsb	r4, r5, r6, rrx         @ encoding: [0x66,0x40,0x65,0xe0]

@ CHECK: rsb	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x65,0xe2]
@ CHECK: rsb	r4, r4, r5              @ encoding: [0x05,0x40,0x64,0xe0]
@ CHECK: rsb	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0x64,0xe0]
@ CHECK: rsb	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x64,0xe0]
@ CHECK: rsbne	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x64,0x10]
@ CHECK: rsb	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0x64,0xe0]
@ CHECK: rsb	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0x64,0xe0]
@ CHECK: rsbgt	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0x66,0xc0]
@ CHECK: rsb	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0x66,0xe0]
@ CHECK: rsb	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0x66,0xe0]
@ CHECK: rsb	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0x66,0xe0]
@ CHECK: rsb	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0x64,0xe0]

@------------------------------------------------------------------------------
@ RSC
@------------------------------------------------------------------------------
        rsc r4, r5, #0xf000
        rsc r4, r5, r6
        rsc r4, r5, r6, lsl #5
        rsclo r4, r5, r6, lsr #5
        rsc r4, r5, r6, lsr #5
        rsc r4, r5, r6, asr #5
        rsc r4, r5, r6, ror #5
        rsc r6, r7, r8, lsl r9
        rsc r6, r7, r8, lsr r9
        rsc r6, r7, r8, asr r9
        rscle r6, r7, r8, ror r9

        @ destination register is optional
        rsc r5, #0xf000
        rsc r4, r5
        rsc r4, r5, lsl #5
        rsc r4, r5, lsr #5
        rscne r4, r5, lsr #5
        rsc r4, r5, asr #5
        rsc r4, r5, ror #5
        rscgt r6, r7, lsl r9
        rsc r6, r7, lsr r9
        rsc r6, r7, asr r9
        rsc r6, r7, ror r9

@ CHECK: rsc	r4, r5, #61440          @ encoding: [0x0f,0x4a,0xe5,0xe2]
@ CHECK: rsc	r4, r5, r6              @ encoding: [0x06,0x40,0xe5,0xe0]
@ CHECK: rsc	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0xe5,0xe0]
@ CHECK: rsclo	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0xe5,0x30]
@ CHECK: rsc	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0xe5,0xe0]
@ CHECK: rsc	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0xe5,0xe0]
@ CHECK: rsc	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0xe5,0xe0]
@ CHECK: rsc	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0xe7,0xe0]
@ CHECK: rsc	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0xe7,0xe0]
@ CHECK: rsc	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0xe7,0xe0]
@ CHECK: rscle	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0xe7,0xd0]

@ CHECK: rsc	r5, r5, #61440          @ encoding: [0x0f,0x5a,0xe5,0xe2]
@ CHECK: rsc	r4, r4, r5              @ encoding: [0x05,0x40,0xe4,0xe0]
@ CHECK: rsc	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0xe4,0xe0]
@ CHECK: rsc	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0xe4,0xe0]
@ CHECK: rscne	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0xe4,0x10]
@ CHECK: rsc	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0xe4,0xe0]
@ CHECK: rsc	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0xe4,0xe0]
@ CHECK: rscgt	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0xe6,0xc0]
@ CHECK: rsc	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0xe6,0xe0]
@ CHECK: rsc	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0xe6,0xe0]
@ CHECK: rsc	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0xe6,0xe0]

@------------------------------------------------------------------------------
@ SADD16/SADD8
@------------------------------------------------------------------------------
        sadd16 r1, r2, r3
        sadd16gt r1, r2, r3
        sadd8 r1, r2, r3
        sadd8le r1, r2, r3

@ CHECK: sadd16	r1, r2, r3              @ encoding: [0x13,0x1f,0x12,0xe6]
@ CHECK: sadd16gt	r1, r2, r3      @ encoding: [0x13,0x1f,0x12,0xc6]
@ CHECK: sadd8	r1, r2, r3              @ encoding: [0x93,0x1f,0x12,0xe6]
@ CHECK: sadd8le r1, r2, r3             @ encoding: [0x93,0x1f,0x12,0xd6]


@------------------------------------------------------------------------------
@ SASX
@------------------------------------------------------------------------------
        sasx r9, r12, r0
        sasxeq r9, r12, r0

@ CHECK: sasx	r9, r12, r0             @ encoding: [0x30,0x9f,0x1c,0xe6]
@ CHECK: sasxeq	r9, r12, r0             @ encoding: [0x30,0x9f,0x1c,0x06]


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
