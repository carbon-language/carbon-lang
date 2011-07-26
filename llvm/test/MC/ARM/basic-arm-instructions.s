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
@ SBC
@------------------------------------------------------------------------------
        sbc r4, r5, #0xf000
        sbc r4, r5, r6
        sbc r4, r5, r6, lsl #5
        sbc r4, r5, r6, lsr #5
        sbc r4, r5, r6, lsr #5
        sbc r4, r5, r6, asr #5
        sbc r4, r5, r6, ror #5
        sbc r6, r7, r8, lsl r9
        sbc r6, r7, r8, lsr r9
        sbc r6, r7, r8, asr r9
        sbc r6, r7, r8, ror r9

        @ destination register is optional
        sbc r5, #0xf000
        sbc r4, r5
        sbc r4, r5, lsl #5
        sbc r4, r5, lsr #5
        sbc r4, r5, lsr #5
        sbc r4, r5, asr #5
        sbc r4, r5, ror #5
        sbc r6, r7, lsl r9
        sbc r6, r7, lsr r9
        sbc r6, r7, asr r9
        sbc r6, r7, ror r9

@ CHECK: sbc	r4, r5, #61440          @ encoding: [0x0f,0x4a,0xc5,0xe2]
@ CHECK: sbc	r4, r5, r6              @ encoding: [0x06,0x40,0xc5,0xe0]
@ CHECK: sbc	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0xc5,0xe0]
@ CHECK: sbc	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0xc5,0xe0]
@ CHECK: sbc	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0xc5,0xe0]
@ CHECK: sbc	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0xc5,0xe0]
@ CHECK: sbc	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0xc5,0xe0]
@ CHECK: sbc	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0xc7,0xe0]
@ CHECK: sbc	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0xc7,0xe0]
@ CHECK: sbc	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0xc7,0xe0]
@ CHECK: sbc	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0xc7,0xe0]

@ CHECK: sbc	r5, r5, #61440          @ encoding: [0x0f,0x5a,0xc5,0xe2]
@ CHECK: sbc	r4, r4, r5              @ encoding: [0x05,0x40,0xc4,0xe0]
@ CHECK: sbc	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0xc4,0xe0]
@ CHECK: sbc	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0xc4,0xe0]
@ CHECK: sbc	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0xc4,0xe0]
@ CHECK: sbc	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0xc4,0xe0]
@ CHECK: sbc	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0xc4,0xe0]
@ CHECK: sbc	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0xc6,0xe0]
@ CHECK: sbc	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0xc6,0xe0]
@ CHECK: sbc	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0xc6,0xe0]
@ CHECK: sbc	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0xc6,0xe0]


@------------------------------------------------------------------------------
@ SEL
@------------------------------------------------------------------------------
        sel r9, r2, r1
        selne r9, r2, r1

@ CHECK: sel	r9, r2, r1              @ encoding: [0xb1,0x9f,0x82,0xe6]
@ CHECK: selne	r9, r2, r1              @ encoding: [0xb1,0x9f,0x82,0x16]


@------------------------------------------------------------------------------
@ SETEND
@------------------------------------------------------------------------------
        setend be
        setend le

	sel	r9, r2, r1              @ encoding: [0xb1,0x9f,0x82,0xe6]
	selne	r9, r2, r1              @ encoding: [0xb1,0x9f,0x82,0x16]


@------------------------------------------------------------------------------
@ SEV
@------------------------------------------------------------------------------
        sev
        seveq

@ CHECK: sev                             @ encoding: [0x04,0xf0,0x20,0xe3]
@ CHECK: seveq                           @ encoding: [0x04,0xf0,0x20,0x03]

@------------------------------------------------------------------------------
@ SHADD16/SHADD8
@------------------------------------------------------------------------------
        shadd16 r4, r8, r2
        shadd16gt r4, r8, r2
        shadd8 r4, r8, r2
        shadd8gt r4, r8, r2

@ CHECK: shadd16	r4, r8, r2      @ encoding: [0x12,0x4f,0x38,0xe6]
@ CHECK: shadd16gt	r4, r8, r2      @ encoding: [0x12,0x4f,0x38,0xc6]
@ CHECK: shadd8	r4, r8, r2              @ encoding: [0x92,0x4f,0x38,0xe6]
@ CHECK: shadd8gt	r4, r8, r2      @ encoding: [0x92,0x4f,0x38,0xc6]


@------------------------------------------------------------------------------
@ SHASX
@------------------------------------------------------------------------------
        shasx r4, r8, r2
        shasxgt r4, r8, r2

@ CHECK: shasx	r4, r8, r2              @ encoding: [0x32,0x4f,0x38,0xe6]
@ CHECK: shasxgt r4, r8, r2             @ encoding: [0x32,0x4f,0x38,0xc6]


@------------------------------------------------------------------------------
@ SHSUB16/SHSUB8
@------------------------------------------------------------------------------
        shsub16 r4, r8, r2
        shsub16gt r4, r8, r2
        shsub8 r4, r8, r2
        shsub8gt r4, r8, r2

@ CHECK: shsub16	r4, r8, r2      @ encoding: [0x72,0x4f,0x38,0xe6]
@ CHECK: shsub16gt	r4, r8, r2      @ encoding: [0x72,0x4f,0x38,0xc6]
@ CHECK: shsub8	r4, r8, r2              @ encoding: [0xf2,0x4f,0x38,0xe6]
@ CHECK: shsub8gt	r4, r8, r2      @ encoding: [0xf2,0x4f,0x38,0xc6]

@------------------------------------------------------------------------------
@ SMC
@------------------------------------------------------------------------------
        smc #0xf
        smceq #0

@ CHECK: smc	#15                     @ encoding: [0x7f,0x00,0x60,0xe1]
@ CHECK: smceq	#0                      @ encoding: [0x70,0x00,0x60,0x01]

@------------------------------------------------------------------------------
@ SMLABB/SMLABT/SMLATB/SMLATT
@------------------------------------------------------------------------------
        smlabb r3, r1, r9, r0
        smlabt r5, r6, r4, r1
        smlatb r4, r2, r3, r2
        smlatt r8, r3, r8, r4
        smlabbge r3, r1, r9, r0
        smlabtle r5, r6, r4, r1
        smlatbne r4, r2, r3, r2
        smlatteq r8, r3, r8, r4

@ CHECK: smlabb	r3, r1, r9, r0          @ encoding: [0x81,0x09,0x03,0xe1]
@ CHECK: smlabt	r5, r6, r4, r1          @ encoding: [0xc6,0x14,0x05,0xe1]
@ CHECK: smlatb	r4, r2, r3, r2          @ encoding: [0xa2,0x23,0x04,0xe1]
@ CHECK: smlatt	r8, r3, r8, r4          @ encoding: [0xe3,0x48,0x08,0xe1]
@ CHECK: smlabbge	r3, r1, r9, r0  @ encoding: [0x81,0x09,0x03,0xa1]
@ CHECK: smlabtle	r5, r6, r4, r1  @ encoding: [0xc6,0x14,0x05,0xd1]
@ CHECK: smlatbne	r4, r2, r3, r2  @ encoding: [0xa2,0x23,0x04,0x11]
@ CHECK: smlatteq	r8, r3, r8, r4  @ encoding: [0xe3,0x48,0x08,0x01]

@------------------------------------------------------------------------------
@ SMLAD/SMLADX
@------------------------------------------------------------------------------
        smlad r2, r3, r5, r8
        smladx r2, r3, r5, r8
        smladeq r2, r3, r5, r8
        smladxhi r2, r3, r5, r8

@ CHECK: smlad	r2, r3, r5, r8          @ encoding: [0x13,0x85,0x02,0xe7]
@ CHECK: smladx	r2, r3, r5, r8          @ encoding: [0x33,0x85,0x02,0xe7]
@ CHECK: smladeq	r2, r3, r5, r8  @ encoding: [0x13,0x85,0x02,0x07]
@ CHECK: smladxhi	r2, r3, r5, r8  @ encoding: [0x33,0x85,0x02,0x87]


@------------------------------------------------------------------------------
@ SMLAL
@------------------------------------------------------------------------------
        smlal r2, r3, r5, r8
        smlals r2, r3, r5, r8
        smlaleq r2, r3, r5, r8
        smlalshi r2, r3, r5, r8

@ CHECK: smlal	r2, r3, r5, r8          @ encoding: [0x95,0x28,0xe3,0xe0]
@ CHECK: smlals	r2, r3, r5, r8          @ encoding: [0x95,0x28,0xf3,0xe0]
@ CHECK: smlaleq	r2, r3, r5, r8  @ encoding: [0x95,0x28,0xe3,0x00]
@ CHECK: smlalshi	r2, r3, r5, r8  @ encoding: [0x95,0x28,0xf3,0x80]


@------------------------------------------------------------------------------
@ SMLALBB/SMLALBT/SMLALTB/SMLALTT
@------------------------------------------------------------------------------
        smlalbb r3, r1, r9, r0
        smlalbt r5, r6, r4, r1
        smlaltb r4, r2, r3, r2
        smlaltt r8, r3, r8, r4
        smlalbbge r3, r1, r9, r0
        smlalbtle r5, r6, r4, r1
        smlaltbne r4, r2, r3, r2
        smlaltteq r8, r3, r8, r4

@ CHECK: smlalbb	r3, r1, r9, r0  @ encoding: [0x89,0x30,0x41,0xe1]
@ CHECK: smlalbt	r5, r6, r4, r1  @ encoding: [0xc4,0x51,0x46,0xe1]
@ CHECK: smlaltb	r4, r2, r3, r2  @ encoding: [0xa3,0x42,0x42,0xe1]
@ CHECK: smlaltt	r8, r3, r8, r4  @ encoding: [0xe8,0x84,0x43,0xe1]
@ CHECK: smlalbbge	r3, r1, r9, r0  @ encoding: [0x89,0x30,0x41,0xa1]
@ CHECK: smlalbtle	r5, r6, r4, r1  @ encoding: [0xc4,0x51,0x46,0xd1]
@ CHECK: smlaltbne	r4, r2, r3, r2  @ encoding: [0xa3,0x42,0x42,0x11]
@ CHECK: smlaltteq	r8, r3, r8, r4  @ encoding: [0xe8,0x84,0x43,0x01]


@------------------------------------------------------------------------------
@ SMLALD/SMLALDX
@------------------------------------------------------------------------------
        smlald r2, r3, r5, r8
        smlaldx r2, r3, r5, r8
        smlaldeq r2, r3, r5, r8
        smlaldxhi r2, r3, r5, r8

@ CHECK: smlald	r2, r3, r5, r8          @ encoding: [0x15,0x28,0x43,0xe7]
@ CHECK: smlaldx	r2, r3, r5, r8  @ encoding: [0x35,0x28,0x43,0xe7]
@ CHECK: smlaldeq	r2, r3, r5, r8  @ encoding: [0x15,0x28,0x43,0x07]
@ CHECK: smlaldxhi	r2, r3, r5, r8  @ encoding: [0x35,0x28,0x43,0x87]


@------------------------------------------------------------------------------
@ SMLAWB/SMLAWT
@------------------------------------------------------------------------------
        smlawb r2, r3, r10, r8
        smlawt r8, r3, r5, r9
        smlawbeq r2, r7, r5, r8
        smlawthi r1, r3, r0, r8

@ CHECK: smlawb	r2, r3, r10, r8         @ encoding: [0x83,0x8a,0x22,0xe1]
@ CHECK: smlawt	r8, r3, r5, r9          @ encoding: [0xc3,0x95,0x28,0xe1]
@ CHECK: smlawbeq	r2, r7, r5, r8  @ encoding: [0x87,0x85,0x22,0x01]
@ CHECK: smlawthi	r1, r3, r0, r8  @ encoding: [0xc3,0x80,0x21,0x81]


@------------------------------------------------------------------------------
@ SMLSD/SMLSDX
@------------------------------------------------------------------------------
        smlsd r2, r3, r5, r8
        smlsdx r2, r3, r5, r8
        smlsdeq r2, r3, r5, r8
        smlsdxhi r2, r3, r5, r8

@ CHECK: smlsd	r2, r3, r5, r8          @ encoding: [0x53,0x85,0x02,0xe7]
@ CHECK: smlsdx	r2, r3, r5, r8          @ encoding: [0x73,0x85,0x02,0xe7]
@ CHECK: smlsdeq	r2, r3, r5, r8  @ encoding: [0x53,0x85,0x02,0x07]
@ CHECK: smlsdxhi	r2, r3, r5, r8  @ encoding: [0x73,0x85,0x02,0x87]


@------------------------------------------------------------------------------
@ SMLSLD/SMLSLDX
@------------------------------------------------------------------------------
        smlsld r2, r9, r5, r1
        smlsldx r4, r11, r2, r8
        smlsldeq r8, r2, r5, r6
        smlsldxhi r1, r0, r3, r8

@ CHECK: smlsld	r2, r9, r5, r1          @ encoding: [0x55,0x21,0x49,0xe7]
@ CHECK: smlsldx	r4, r11, r2, r8 @ encoding: [0x72,0x48,0x4b,0xe7]
@ CHECK: smlsldeq	r8, r2, r5, r6  @ encoding: [0x55,0x86,0x42,0x07]
@ CHECK: smlsldxhi	r1, r0, r3, r8  @ encoding: [0x73,0x18,0x40,0x87]


@------------------------------------------------------------------------------
@ SMMLA/SMMLAR
@------------------------------------------------------------------------------
        smmla r1, r2, r3, r4
        smmlar r4, r3, r2, r1
        smmlalo r1, r2, r3, r4
        smmlarcs r4, r3, r2, r1

@ CHECK: smmla	r1, r2, r3, r4          @ encoding: [0x12,0x43,0x51,0xe7]
@ CHECK: smmlar	r4, r3, r2, r1          @ encoding: [0x33,0x12,0x54,0xe7]
@ CHECK: smmlalo	r1, r2, r3, r4  @ encoding: [0x12,0x43,0x51,0x37]
@ CHECK: smmlarhs	r4, r3, r2, r1  @ encoding: [0x33,0x12,0x54,0x27]


@------------------------------------------------------------------------------
@ SMMLS/SMMLSR
@------------------------------------------------------------------------------
        smmls r1, r2, r3, r4
        smmlsr r4, r3, r2, r1
        smmlslo r1, r2, r3, r4
        smmlsrcs r4, r3, r2, r1

@ CHECK: smmls	r1, r2, r3, r4          @ encoding: [0xd2,0x43,0x51,0xe7]
@ CHECK: smmlsr	r4, r3, r2, r1          @ encoding: [0xf3,0x12,0x54,0xe7]
@ CHECK: smmlslo	r1, r2, r3, r4  @ encoding: [0xd2,0x43,0x51,0x37]
@ CHECK: smmlsrhs	r4, r3, r2, r1  @ encoding: [0xf3,0x12,0x54,0x27]


@------------------------------------------------------------------------------
@ SMMUL/SMMULR
@------------------------------------------------------------------------------
        smmul r2, r3, r4
        smmulr r3, r2, r1
        smmulcc r2, r3, r4
        smmulrhs r3, r2, r1

@ CHECK: smmul	r2, r3, r4              @ encoding: [0x13,0xf4,0x52,0xe7]
@ CHECK: smmulr	r3, r2, r1              @ encoding: [0x32,0xf1,0x53,0xe7]
@ CHECK: smmullo	r2, r3, r4      @ encoding: [0x13,0xf4,0x52,0x37]
@ CHECK: smmulrhs	r3, r2, r1      @ encoding: [0x32,0xf1,0x53,0x27]


@------------------------------------------------------------------------------
@ SMUAD/SMUADX
@------------------------------------------------------------------------------
        smuad r2, r3, r4
        smuadx r3, r2, r1
        smuadlt r2, r3, r4
        smuadxge r3, r2, r1

@ CHECK: smuad	r2, r3, r4              @ encoding: [0x13,0xf4,0x02,0xe7]
@ CHECK: smuadx	r3, r2, r1              @ encoding: [0x32,0xf1,0x03,0xe7]
@ CHECK: smuadlt	r2, r3, r4      @ encoding: [0x13,0xf4,0x02,0xb7]
@ CHECK: smuadxge	r3, r2, r1      @ encoding: [0x32,0xf1,0x03,0xa7]


@------------------------------------------------------------------------------
@ SMULBB/SMLALBT/SMLALTB/SMLALTT
@------------------------------------------------------------------------------
        smulbb r3, r9, r0
        smulbt r5, r4, r1
        smultb r4, r2, r2
        smultt r8, r3, r4
        smulbbge r1, r9, r0
        smulbtle r5, r6, r4
        smultbne r2, r3, r2
        smultteq r8, r3, r4

@ CHECK: smulbb	r3, r9, r0              @ encoding: [0x89,0x00,0x63,0xe1]
@ CHECK: smulbt	r5, r4, r1              @ encoding: [0xc4,0x01,0x65,0xe1]
@ CHECK: smultb	r4, r2, r2              @ encoding: [0xa2,0x02,0x64,0xe1]
@ CHECK: smultt	r8, r3, r4              @ encoding: [0xe3,0x04,0x68,0xe1]
@ CHECK: smulbbge	r1, r9, r0      @ encoding: [0x89,0x00,0x61,0xa1]
@ CHECK: smulbtle	r5, r6, r4      @ encoding: [0xc6,0x04,0x65,0xd1]
@ CHECK: smultbne	r2, r3, r2      @ encoding: [0xa3,0x02,0x62,0x11]
@ CHECK: smultteq	r8, r3, r4      @ encoding: [0xe3,0x04,0x68,0x01]


@------------------------------------------------------------------------------
@ SMULL
@------------------------------------------------------------------------------
        smull r3, r9, r0, r1
        smulls r3, r9, r0, r2
        smulleq r8, r3, r4, r5
        smullseq r8, r3, r4, r3

@ CHECK: smull	r3, r9, r0, r1          @ encoding: [0x90,0x31,0xc9,0xe0]
@ CHECK: smulls	r3, r9, r0, r2          @ encoding: [0x90,0x32,0xd9,0xe0]
@ CHECK: smulleq	r8, r3, r4, r5  @ encoding: [0x94,0x85,0xc3,0x00]
@ CHECK: smullseq	r8, r3, r4, r3  @ encoding: [0x94,0x83,0xd3,0x00]


@------------------------------------------------------------------------------
@ SMULWB/SMULWT
@------------------------------------------------------------------------------
        smulwb r3, r9, r0
        smulwt r3, r9, r2

@ CHECK: smulwb	r3, r9, r0              @ encoding: [0xa9,0x00,0x23,0xe1]
@ CHECK: smulwt	r3, r9, r2              @ encoding: [0xe9,0x02,0x23,0xe1]


@------------------------------------------------------------------------------
@ SMUSD/SMUSDX
@------------------------------------------------------------------------------
        smusd r3, r0, r1
        smusdx r3, r9, r2
        smusdeq r8, r3, r2
        smusdxne r7, r4, r3

@ CHECK: smusd	r3, r0, r1              @ encoding: [0x50,0xf1,0x03,0xe7]
@ CHECK: smusdx	r3, r9, r2              @ encoding: [0x79,0xf2,0x03,0xe7]
@ CHECK: smusdeq	r8, r3, r2      @ encoding: [0x53,0xf2,0x08,0x07]
@ CHECK: smusdxne	r7, r4, r3      @ encoding: [0x74,0xf3,0x07,0x17]


@------------------------------------------------------------------------------
@ FIXME: SRS
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ SSAT
@------------------------------------------------------------------------------
	ssat	r8, #1, r10
	ssat	r8, #1, r10, lsl #0
	ssat	r8, #1, r10, lsl #31
	ssat	r8, #1, r10, asr #32
	ssat	r8, #1, r10, asr #1

@ CHECK: ssat	r8, #1, r10             @ encoding: [0x1a,0x80,0xa0,0xe6]
@ CHECK: ssat	r8, #1, r10             @ encoding: [0x1a,0x80,0xa0,0xe6]
@ CHECK: ssat	r8, #1, r10, lsl #31    @ encoding: [0x9a,0x8f,0xa0,0xe6]
@ CHECK: ssat	r8, #1, r10, asr #32    @ encoding: [0x5a,0x80,0xa0,0xe6]
@ CHECK: ssat	r8, #1, r10, asr #1     @ encoding: [0xda,0x80,0xa0,0xe6]


@------------------------------------------------------------------------------
@ SSAT16
@------------------------------------------------------------------------------
	ssat16	r2, #1, r7
	ssat16	r3, #16, r5

@ CHECK: ssat16	r2, #1, r7              @ encoding: [0x37,0x2f,0xa0,0xe6]
@ CHECK: ssat16	r3, #16, r5             @ encoding: [0x35,0x3f,0xaf,0xe6]


@------------------------------------------------------------------------------
@ SSAX
@------------------------------------------------------------------------------
        ssax r2, r3, r4
        ssaxlt r2, r3, r4

@ CHECK: ssax	r2, r3, r4              @ encoding: [0x54,0x2f,0x13,0xe6]
@ CHECK: ssaxlt	r2, r3, r4              @ encoding: [0x54,0x2f,0x13,0xb6]

@------------------------------------------------------------------------------
@ SSUB16/SSUB8
@------------------------------------------------------------------------------
        ssub16 r1, r0, r6
        ssub16ne r5, r3, r2
        ssub8 r9, r2, r4
        ssub8eq r5, r1, r2

@ CHECK: ssub16	r1, r0, r6              @ encoding: [0x76,0x1f,0x10,0xe6]
@ CHECK: ssub16ne	r5, r3, r2      @ encoding: [0x72,0x5f,0x13,0x16]
@ CHECK: ssub8	r9, r2, r4              @ encoding: [0xf4,0x9f,0x12,0xe6]
@ CHECK: ssub8eq	r5, r1, r2      @ encoding: [0xf2,0x5f,0x11,0x06]


@------------------------------------------------------------------------------
@ STM*
@------------------------------------------------------------------------------
        stm       r2, {r1,r3-r6,sp}
        stmia     r3, {r1,r3-r6,lr}
        stmib     r4, {r1,r3-r6,sp}
        stmda     r5, {r1,r3-r6,sp}
        stmdb     r6, {r1,r3-r6,r8}
        stmfd     sp, {r1,r3-r6,sp}

        @ with update
        stm       r8!, {r1,r3-r6,sp}
        stmib     r9!, {r1,r3-r6,sp}
        stmda     sp!, {r1,r3-r6}
        stmdb     r0!, {r1,r5,r7,sp}

@ CHECK: stm	r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x82,0xe8]
@ CHECK: stm	r3, {lr, r1, r3, r4, r5, r6} @ encoding: [0x7a,0x40,0x83,0xe8]
@ CHECK: stmib	r4, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x84,0xe9]
@ CHECK: stmda	r5, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x05,0xe8]
@ CHECK: stmdb	r6, {r1, r3, r4, r5, r6, r8} @ encoding: [0x7a,0x01,0x06,0xe9]
@ CHECK: stmdb	sp, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x0d,0xe9]

@ CHECK: stm	r8!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xa8,0xe8]
@ CHECK: stmib	r9!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xa9,0xe9]
@ CHECK: stmda	sp!, {r1, r3, r4, r5, r6} @ encoding: [0x7a,0x00,0x2d,0xe8]
@ CHECK: stmdb	r0!, {r1, r5, r7, sp}   @ encoding: [0xa2,0x20,0x20,0xe9]


@------------------------------------------------------------------------------
@ FIXME:STR*
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ SUB
@------------------------------------------------------------------------------
        sub r4, r5, #0xf000
        sub r4, r5, r6
        sub r4, r5, r6, lsl #5
        sub r4, r5, r6, lsr #5
        sub r4, r5, r6, lsr #5
        sub r4, r5, r6, asr #5
        sub r4, r5, r6, ror #5
        sub r6, r7, r8, lsl r9
        sub r6, r7, r8, lsr r9
        sub r6, r7, r8, asr r9
        sub r6, r7, r8, ror r9

        @ destination register is optional
        sub r5, #0xf000
        sub r4, r5
        sub r4, r5, lsl #5
        sub r4, r5, lsr #5
        sub r4, r5, lsr #5
        sub r4, r5, asr #5
        sub r4, r5, ror #5
        sub r6, r7, lsl r9
        sub r6, r7, lsr r9
        sub r6, r7, asr r9
        sub r6, r7, ror r9

@ CHECK: sub	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x45,0xe2]
@ CHECK: sub	r4, r5, r6              @ encoding: [0x06,0x40,0x45,0xe0]
@ CHECK: sub	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0x45,0xe0]
@ CHECK: sub	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x45,0xe0]
@ CHECK: sub	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x45,0xe0]
@ CHECK: sub	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0x45,0xe0]
@ CHECK: sub	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0x45,0xe0]
@ CHECK: sub	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0x47,0xe0]
@ CHECK: sub	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0x47,0xe0]
@ CHECK: sub	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0x47,0xe0]
@ CHECK: sub	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0x47,0xe0]


@ CHECK: sub	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x45,0xe2]
@ CHECK: sub	r4, r4, r5              @ encoding: [0x05,0x40,0x44,0xe0]
@ CHECK: sub	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0x44,0xe0]
@ CHECK: sub	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x44,0xe0]
@ CHECK: sub	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x44,0xe0]
@ CHECK: sub	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0x44,0xe0]
@ CHECK: sub	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0x44,0xe0]
@ CHECK: sub	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0x46,0xe0]
@ CHECK: sub	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0x46,0xe0]
@ CHECK: sub	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0x46,0xe0]
@ CHECK: sub	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0x46,0xe0]


@------------------------------------------------------------------------------
@ SVC
@------------------------------------------------------------------------------
        svc #16
        svc #0
        svc #0xffffff

@ CHECK: svc	#16                     @ encoding: [0x10,0x00,0x00,0xef]
@ CHECK: svc	#0                      @ encoding: [0x00,0x00,0x00,0xef]
@ CHECK: svc	#16777215               @ encoding: [0xff,0xff,0xff,0xef]
