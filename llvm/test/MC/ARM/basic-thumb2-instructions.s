@ RUN: llvm-mc -triple=thumbv7-apple-darwin -show-encoding < %s | FileCheck %s
  .syntax unified
  .globl _func

@ Check that the assembler can handle the documented syntax from the ARM ARM.
@ For complex constructs like shifter operands, check more thoroughly for them
@ once then spot check that following instructions accept the form generally.
@ This gives us good coverage while keeping the overall size of the test
@ more reasonable.


@ FIXME: Some 3-operand instructions have a 2-operand assembly syntax.

_func:
@ CHECK: _func

@------------------------------------------------------------------------------
@ ADC (immediate)
@------------------------------------------------------------------------------
        adc r0, r1, #4
        adcs r0, r1, #0
        adc r1, r2, #255
        adc r3, r7, #0x00550055
        adc r8, r12, #0xaa00aa00
        adc r9, r7, #0xa5a5a5a5
        adc r5, r3, #0x87000000
        adc r4, r2, #0x7f800000
        adc r4, r2, #0x00000680

@ CHECK: adc	r0, r1, #4              @ encoding: [0x41,0xf1,0x04,0x00]
@ CHECK: adcs	r0, r1, #0              @ encoding: [0x51,0xf1,0x00,0x00]
@ CHECK: adc	r1, r2, #255            @ encoding: [0x42,0xf1,0xff,0x01]
@ CHECK: adc	r3, r7, #5570645        @ encoding: [0x47,0xf1,0x55,0x13]
@ CHECK: adc	r8, r12, #2852170240    @ encoding: [0x4c,0xf1,0xaa,0x28]
@ CHECK: adc	r9, r7, #2779096485     @ encoding: [0x47,0xf1,0xa5,0x39]
@ CHECK: adc	r5, r3, #2264924160     @ encoding: [0x43,0xf1,0x07,0x45]
@ CHECK: adc	r4, r2, #2139095040     @ encoding: [0x42,0xf1,0xff,0x44]
@ CHECK: adc	r4, r2, #1664           @ encoding: [0x42,0xf5,0xd0,0x64]

@------------------------------------------------------------------------------
@ ADC (register)
@------------------------------------------------------------------------------
        adc r4, r5, r6
        adcs r4, r5, r6
        adc.w r9, r1, r3
        adcs.w r9, r1, r3
        adc	r0, r1, r3, ror #4
        adcs	r0, r1, r3, lsl #7
        adc.w	r0, r1, r3, lsr #31
        adcs.w	r0, r1, r3, asr #32

@ CHECK: adc.w	r4, r5, r6              @ encoding: [0x45,0xeb,0x06,0x04]
@ CHECK: adcs.w	r4, r5, r6              @ encoding: [0x55,0xeb,0x06,0x04]
@ CHECK: adc.w	r9, r1, r3              @ encoding: [0x41,0xeb,0x03,0x09]
@ CHECK: adcs.w	r9, r1, r3              @ encoding: [0x51,0xeb,0x03,0x09]
@ CHECK: adc.w	r0, r1, r3, ror #4      @ encoding: [0x41,0xeb,0x33,0x10]
@ CHECK: adcs.w	r0, r1, r3, lsl #7      @ encoding: [0x51,0xeb,0xc3,0x10]
@ CHECK: adc.w	r0, r1, r3, lsr #31     @ encoding: [0x41,0xeb,0xd3,0x70]
@ CHECK: adcs.w	r0, r1, r3, asr #32     @ encoding: [0x51,0xeb,0x23,0x00]


@------------------------------------------------------------------------------
@ ADD (immediate)
@------------------------------------------------------------------------------
        itet eq
        addeq r1, r2, #4
        addwne r5, r3, #1023
        addeq r4, r5, #293
        add r2, sp, #1024
        add r2, r8, #0xff00
        add r2, r3, #257
        addw r2, r3, #257
        add r12, r6, #0x100
        addw r12, r6, #0x100
        adds r1, r2, #0x1f0

@ CHECK: itet	eq                      @ encoding: [0x0a,0xbf]
@ CHECK: addeq	r1, r2, #4              @ encoding: [0x11,0x1d]
@ CHECK: addwne	r5, r3, #1023           @ encoding: [0x03,0xf2,0xff,0x35]
@ CHECK: addweq	r4, r5, #293            @ encoding: [0x05,0xf2,0x25,0x14]
@ CHECK: add.w	r2, sp, #1024           @ encoding: [0x0d,0xf5,0x80,0x62]
@ CHECK: add.w	r2, r8, #65280          @ encoding: [0x08,0xf5,0x7f,0x42]
@ CHECK: addw	r2, r3, #257            @ encoding: [0x03,0xf2,0x01,0x12]
@ CHECK: addw	r2, r3, #257            @ encoding: [0x03,0xf2,0x01,0x12]
@ CHECK: add.w	r12, r6, #256           @ encoding: [0x06,0xf5,0x80,0x7c]
@ CHECK: addw	r12, r6, #256           @ encoding: [0x06,0xf2,0x00,0x1c]
@ CHECK: adds.w	r1, r2, #496            @ encoding: [0x12,0xf5,0xf8,0x71]


@------------------------------------------------------------------------------
@ ADD (register)
@------------------------------------------------------------------------------
        add r1, r2, r8
        add r5, r9, r2, asr #32
        adds r7, r3, r1, lsl #31
        adds.w r0, r3, r6, lsr #25
        add.w r4, r8, r1, ror #12

@ CHECK: add.w	r1, r2, r8              @ encoding: [0x02,0xeb,0x08,0x01]
@ CHECK: add.w	r5, r9, r2, asr #32     @ encoding: [0x09,0xeb,0x22,0x05]
@ CHECK: adds.w	r7, r3, r1, lsl #31     @ encoding: [0x13,0xeb,0xc1,0x77]
@ CHECK: adds.w	r0, r3, r6, lsr #25     @ encoding: [0x13,0xeb,0x56,0x60]
@ CHECK: add.w	r4, r8, r1, ror #12     @ encoding: [0x08,0xeb,0x31,0x34]


@------------------------------------------------------------------------------
@ FIXME: ADR
@------------------------------------------------------------------------------

@------------------------------------------------------------------------------
@ AND (immediate)
@------------------------------------------------------------------------------
        and r2, r5, #0xff000
        ands r3, r12, #0xf
        and r1, #0xff
        and r1, r1, #0xff

@ CHECK: and	r2, r5, #1044480        @ encoding: [0x05,0xf4,0x7f,0x22]
@ CHECK: ands	r3, r12, #15            @ encoding: [0x1c,0xf0,0x0f,0x03]
@ CHECK: and	r1, r1, #255            @ encoding: [0x01,0xf0,0xff,0x01]
@ CHECK: and	r1, r1, #255            @ encoding: [0x01,0xf0,0xff,0x01]


@------------------------------------------------------------------------------
@ AND (register)
@------------------------------------------------------------------------------
        and r4, r9, r8
        and r1, r4, r8, asr #3
        ands r2, r1, r7, lsl #1
        ands.w r4, r5, r2, lsr #20
        and.w r9, r12, r1, ror #17

@ CHECK: and.w	r4, r9, r8              @ encoding: [0x09,0xea,0x08,0x04]
@ CHECK: and.w	r1, r4, r8, asr #3      @ encoding: [0x04,0xea,0xe8,0x01]
@ CHECK: ands.w	r2, r1, r7, lsl #1      @ encoding: [0x11,0xea,0x47,0x02]
@ CHECK: ands.w	r4, r5, r2, lsr #20     @ encoding: [0x15,0xea,0x12,0x54]
@ CHECK: and.w	r9, r12, r1, ror #17    @ encoding: [0x0c,0xea,0x71,0x49]

@------------------------------------------------------------------------------
@ ASR (immediate)
@------------------------------------------------------------------------------
        asr r2, r3, #12
        asrs r8, r3, #32
        asrs.w r2, r3, #1
        asr r2, r3, #4
        asrs r2, r12, #15

        asr r3, #19
        asrs r8, #2
        asrs.w r7, #5
        asr.w r12, #21

@ CHECK: asr.w	r2, r3, #12             @ encoding: [0x4f,0xea,0x23,0x32]
@ CHECK: asrs.w	r8, r3, #32             @ encoding: [0x5f,0xea,0x23,0x08]
@ CHECK: asrs.w	r2, r3, #1              @ encoding: [0x5f,0xea,0x63,0x02]
@ CHECK: asr.w	r2, r3, #4              @ encoding: [0x4f,0xea,0x23,0x12]
@ CHECK: asrs.w	r2, r12, #15            @ encoding: [0x5f,0xea,0xec,0x32]

@ CHECK: asr.w	r3, r3, #19             @ encoding: [0x4f,0xea,0xe3,0x43]
@ CHECK: asrs.w	r8, r8, #2              @ encoding: [0x5f,0xea,0xa8,0x08]
@ CHECK: asrs.w	r7, r7, #5              @ encoding: [0x5f,0xea,0x67,0x17]
@ CHECK: asr.w	r12, r12, #21           @ encoding: [0x4f,0xea,0x6c,0x5c]


@------------------------------------------------------------------------------
@ ASR (register)
@------------------------------------------------------------------------------
        asr r3, r4, r2
        asr.w r1, r2
        asrs r3, r4, r8

@ CHECK: asr.w	r3, r4, r2              @ encoding: [0x44,0xfa,0x02,0xf3]
@ CHECK: asr.w	r1, r1, r2              @ encoding: [0x41,0xfa,0x02,0xf1]
@ CHECK: asrs.w	r3, r4, r8              @ encoding: [0x54,0xfa,0x08,0xf3]


@------------------------------------------------------------------------------
@ B
@------------------------------------------------------------------------------
        b.w   _bar
        beq.w   _bar
        it eq
        beq.w _bar
        bmi.w   #-183396

@ CHECK: b.w	_bar                    @ encoding: [A,0xf0'A',A,0x90'A']
          @   fixup A - offset: 0, value: _bar, kind: fixup_t2_uncondbranch
@ CHECK: beq.w	_bar                    @ encoding: [A,0xf0'A',A,0x80'A']
          @   fixup A - offset: 0, value: _bar, kind: fixup_t2_condbranch
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: b.w	_bar                    @ encoding: [A,0xf0'A',A,0x90'A']
          @   fixup A - offset: 0, value: _bar, kind: fixup_t2_uncondbranch
@ CHECK: bmi.w   #-183396                @ encoding: [0x13,0xf5,0xce,0xa9]

@------------------------------------------------------------------------------
@ CBZ/CBNZ
@------------------------------------------------------------------------------
        cbnz    r7, #6
        cbnz    r7, #12

@ CHECK: cbnz    r7, #6                  @ encoding: [0x1f,0xb9]
@ CHECK: cbnz    r7, #12                 @ encoding: [0x37,0xb9]

@------------------------------------------------------------------------------
@ IT
@------------------------------------------------------------------------------
@ Test encodings of a few full IT blocks, not just the IT instruction

        iteet eq
        addeq r0, r1, r2
        nopne
        subne r5, r6, r7
        addeq r1, r2, #4

@ CHECK: iteet	eq                      @ encoding: [0x0d,0xbf]
@ CHECK: addeq	r0, r1, r2              @ encoding: [0x88,0x18]
@ CHECK: nopne                          @ encoding: [0x00,0xbf]
@ CHECK: subne	r5, r6, r7              @ encoding: [0xf5,0x1b]
@ CHECK: addeq	r1, r2, #4              @ encoding: [0x11,0x1d]
