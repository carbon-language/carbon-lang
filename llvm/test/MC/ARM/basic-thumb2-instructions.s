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
@ BFC
@------------------------------------------------------------------------------
        bfc r5, #3, #17
        it lo
        bfccc r5, #3, #17

@ CHECK: bfc	r5, #3, #17             @ encoding: [0x6f,0xf3,0xd3,0x05]
@ CHECK: it	lo                      @ encoding: [0x38,0xbf]
@ CHECK: bfclo	r5, #3, #17             @ encoding: [0x6f,0xf3,0xd3,0x05]


@------------------------------------------------------------------------------
@ BFI
@------------------------------------------------------------------------------
        bfi r5, r2, #3, #17
        it ne
        bfine r5, r2, #3, #17

@ CHECK: bfi	r5, r2, #3, #17         @ encoding: [0x62,0xf3,0xd3,0x05]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: bfine	r5, r2, #3, #17         @ encoding: [0x62,0xf3,0xd3,0x05]


@------------------------------------------------------------------------------
@ BIC
@------------------------------------------------------------------------------
        bic r10, r1, #0xf
        bic r12, r3, r6
        bic r11, r2, r6, lsl #12
        bic r8, r4, r1, lsr #11
        bic r7, r5, r7, lsr #15
        bic r6, r7, r9, asr #32
        bic r5, r6, r8, ror #1

        @ destination register is optional
        bic r1, #0xf
        bic r1, r1
        bic r4, r2, lsl #31
        bic r6, r3, lsr #12
        bic r7, r4, lsr #7
        bic r8, r5, asr #15
        bic r12, r6, ror #29

@ CHECK: bic	r10, r1, #15            @ encoding: [0x21,0xf0,0x0f,0x0a]
@ CHECK: bic.w	r12, r3, r6             @ encoding: [0x23,0xea,0x06,0x0c]
@ CHECK: bic.w	r11, r2, r6, lsl #12    @ encoding: [0x22,0xea,0x06,0x3b]
@ CHECK: bic.w	r8, r4, r1, lsr #11     @ encoding: [0x24,0xea,0xd1,0x28]
@ CHECK: bic.w	r7, r5, r7, lsr #15     @ encoding: [0x25,0xea,0xd7,0x37]
@ CHECK: bic.w	r6, r7, r9, asr #32     @ encoding: [0x27,0xea,0x29,0x06]
@ CHECK: bic.w	r5, r6, r8, ror #1      @ encoding: [0x26,0xea,0x78,0x05]

@ CHECK: bic	r1, r1, #15             @ encoding: [0x21,0xf0,0x0f,0x01]
@ CHECK: bic.w	r1, r1, r1              @ encoding: [0x21,0xea,0x01,0x01]
@ CHECK: bic.w	r4, r4, r2, lsl #31     @ encoding: [0x24,0xea,0xc2,0x74]
@ CHECK: bic.w	r6, r6, r3, lsr #12     @ encoding: [0x26,0xea,0x13,0x36]
@ CHECK: bic.w	r7, r7, r4, lsr #7      @ encoding: [0x27,0xea,0xd4,0x17]
@ CHECK: bic.w	r8, r8, r5, asr #15     @ encoding: [0x28,0xea,0xe5,0x38]
@ CHECK: bic.w	r12, r12, r6, ror #29   @ encoding: [0x2c,0xea,0x76,0x7c]


@------------------------------------------------------------------------------
@ BXJ
@------------------------------------------------------------------------------
        bxj r5
        it ne
        bxjne r7

@ CHECK: bxj	r5                      @ encoding: [0xc5,0xf3,0x00,0x8f]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: bxjne	r7                      @ encoding: [0xc7,0xf3,0x00,0x8f]


@------------------------------------------------------------------------------
@ CBZ/CBNZ
@------------------------------------------------------------------------------
        cbnz    r7, #6
        cbnz    r7, #12
        cbz   r6, _bar
        cbnz   r6, _bar

@ CHECK: cbnz    r7, #6                  @ encoding: [0x1f,0xb9]
@ CHECK: cbnz    r7, #12                 @ encoding: [0x37,0xb9]
@ CHECK: cbz	r6, _bar                @ encoding: [0x06'A',0xb1'A']
           @   fixup A - offset: 0, value: _bar, kind: fixup_arm_thumb_cb
@ CHECK: cbnz	r6, _bar                @ encoding: [0x06'A',0xb9'A']
           @   fixup A - offset: 0, value: _bar, kind: fixup_arm_thumb_cb


@------------------------------------------------------------------------------
@ CDP/CDP2
@------------------------------------------------------------------------------
  cdp  p7, #1, c1, c1, c1, #4
  cdp2  p7, #1, c1, c1, c1, #4

@ CHECK: cdp	p7, #1, c1, c1, c1, #4  @ encoding: [0x11,0xee,0x81,0x17]
@ CHECK: cdp2	p7, #1, c1, c1, c1, #4  @ encoding: [0x11,0xfe,0x81,0x17]


@------------------------------------------------------------------------------
@ CLREX
@------------------------------------------------------------------------------
        clrex
        it ne
        clrexne

@ CHECK: clrex                           @ encoding: [0xbf,0xf3,0x2f,0x8f]
@ CHECK: it	ne                       @ encoding: [0x18,0xbf]
@ CHECK: clrexne                         @ encoding: [0xbf,0xf3,0x2f,0x8f]


@------------------------------------------------------------------------------
@ CLZ
@------------------------------------------------------------------------------
        clz r1, r2
        it eq
        clzeq r1, r2

@ CHECK: clz	r1, r2                  @ encoding: [0xb2,0xfa,0x82,0xf1]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: clzeq	r1, r2                  @ encoding: [0xb2,0xfa,0x82,0xf1]


@------------------------------------------------------------------------------
@ CMN
@------------------------------------------------------------------------------
        cmn r1, #0xf
        cmn r8, r6
        cmn r1, r6, lsl #10
        cmn r1, r6, lsr #10
        cmn sp, r6, lsr #10
        cmn r1, r6, asr #10
        cmn r1, r6, ror #10

@ CHECK: cmn.w	r1, #15                 @ encoding: [0x11,0xf1,0x0f,0x0f]
@ CHECK: cmn.w	r8, r6                  @ encoding: [0x18,0xeb,0x06,0x0f]
@ CHECK: cmn.w	r1, r6, lsl #10         @ encoding: [0x11,0xeb,0x86,0x2f]
@ CHECK: cmn.w	r1, r6, lsr #10         @ encoding: [0x11,0xeb,0x96,0x2f]
@ CHECK: cmn.w	sp, r6, lsr #10         @ encoding: [0x1d,0xeb,0x96,0x2f]
@ CHECK: cmn.w	r1, r6, asr #10         @ encoding: [0x11,0xeb,0xa6,0x2f]
@ CHECK: cmn.w	r1, r6, ror #10         @ encoding: [0x11,0xeb,0xb6,0x2f]


@------------------------------------------------------------------------------
@ CMP
@------------------------------------------------------------------------------
        cmp r5, #0xff00
        cmp.w r4, r12
        cmp r9, r6, lsl #12
        cmp r3, r7, lsr #31
        cmp sp, r6, lsr #1
        cmp r2, r5, asr #24
        cmp r1, r4, ror #15

@ CHECK: cmp.w	r5, #65280              @ encoding: [0xb5,0xf5,0x7f,0x4f]
@ CHECK: cmp.w	r4, r12                 @ encoding: [0xb4,0xeb,0x0c,0x0f]
@ CHECK: cmp.w	r9, r6, lsl #12         @ encoding: [0xb9,0xeb,0x06,0x3f]
@ CHECK: cmp.w	r3, r7, lsr #31         @ encoding: [0xb3,0xeb,0xd7,0x7f]
@ CHECK: cmp.w	sp, r6, lsr #1          @ encoding: [0xbd,0xeb,0x56,0x0f]
@ CHECK: cmp.w	r2, r5, asr #24         @ encoding: [0xb2,0xeb,0x25,0x6f]
@ CHECK: cmp.w	r1, r4, ror #15         @ encoding: [0xb1,0xeb,0xf4,0x3f]


@------------------------------------------------------------------------------
@ DBG
@------------------------------------------------------------------------------
        dbg #5
        dbg #0
        dbg #15

@ CHECK: dbg	#5                      @ encoding: [0xaf,0xf3,0xf5,0x80]
@ CHECK: dbg	#0                      @ encoding: [0xaf,0xf3,0xf0,0x80]
@ CHECK: dbg	#15                     @ encoding: [0xaf,0xf3,0xff,0x80]


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

@ CHECK: dmb	sy                      @ encoding: [0xbf,0xf3,0x5f,0x8f]
@ CHECK: dmb	st                      @ encoding: [0xbf,0xf3,0x5e,0x8f]
@ CHECK: dmb	ish                     @ encoding: [0xbf,0xf3,0x5b,0x8f]
@ CHECK: dmb	ish                     @ encoding: [0xbf,0xf3,0x5b,0x8f]
@ CHECK: dmb	ishst                   @ encoding: [0xbf,0xf3,0x5a,0x8f]
@ CHECK: dmb	ishst                   @ encoding: [0xbf,0xf3,0x5a,0x8f]
@ CHECK: dmb	nsh                     @ encoding: [0xbf,0xf3,0x57,0x8f]
@ CHECK: dmb	nsh                     @ encoding: [0xbf,0xf3,0x57,0x8f]
@ CHECK: dmb	nshst                   @ encoding: [0xbf,0xf3,0x56,0x8f]
@ CHECK: dmb	nshst                   @ encoding: [0xbf,0xf3,0x56,0x8f]
@ CHECK: dmb	osh                     @ encoding: [0xbf,0xf3,0x53,0x8f]
@ CHECK: dmb	oshst                   @ encoding: [0xbf,0xf3,0x52,0x8f]
@ CHECK: dmb	sy                      @ encoding: [0xbf,0xf3,0x5f,0x8f]


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

@ CHECK: dsb	sy                      @ encoding: [0xbf,0xf3,0x4f,0x8f]
@ CHECK: dsb	st                      @ encoding: [0xbf,0xf3,0x4e,0x8f]
@ CHECK: dsb	ish                     @ encoding: [0xbf,0xf3,0x4b,0x8f]
@ CHECK: dsb	ish                     @ encoding: [0xbf,0xf3,0x4b,0x8f]
@ CHECK: dsb	ishst                   @ encoding: [0xbf,0xf3,0x4a,0x8f]
@ CHECK: dsb	ishst                   @ encoding: [0xbf,0xf3,0x4a,0x8f]
@ CHECK: dsb	nsh                     @ encoding: [0xbf,0xf3,0x47,0x8f]
@ CHECK: dsb	nsh                     @ encoding: [0xbf,0xf3,0x47,0x8f]
@ CHECK: dsb	nshst                   @ encoding: [0xbf,0xf3,0x46,0x8f]
@ CHECK: dsb	nshst                   @ encoding: [0xbf,0xf3,0x46,0x8f]
@ CHECK: dsb	osh                     @ encoding: [0xbf,0xf3,0x43,0x8f]
@ CHECK: dsb	oshst                   @ encoding: [0xbf,0xf3,0x42,0x8f]
@ CHECK: dsb	sy                      @ encoding: [0xbf,0xf3,0x4f,0x8f]


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

@ CHECK: eor	r4, r5, #61440          @ encoding: [0x85,0xf4,0x70,0x44]
@ CHECK: eor.w	r4, r5, r6              @ encoding: [0x85,0xea,0x06,0x04]
@ CHECK: eor.w	r4, r5, r6, lsl #5      @ encoding: [0x85,0xea,0x46,0x14]
@ CHECK: eor.w	r4, r5, r6, lsr #5      @ encoding: [0x85,0xea,0x56,0x14]
@ CHECK: eor.w	r4, r5, r6, lsr #5      @ encoding: [0x85,0xea,0x56,0x14]
@ CHECK: eor.w	r4, r5, r6, asr #5      @ encoding: [0x85,0xea,0x66,0x14]
@ CHECK: eor.w	r4, r5, r6, ror #5      @ encoding: [0x85,0xea,0x76,0x14]


@------------------------------------------------------------------------------
@ ISB
@------------------------------------------------------------------------------
        isb sy
        isb

@ CHECK: isb	sy                      @ encoding: [0xbf,0xf3,0x6f,0x8f]
@ CHECK: isb	sy                      @ encoding: [0xbf,0xf3,0x6f,0x8f]


@------------------------------------------------------------------------------
@ LDMIA
@------------------------------------------------------------------------------
        ldmia.w r4, {r4, r5, r8, r9}
        ldmia.w r4, {r5, r6}
        ldmia.w r5!, {r3, r8}
        ldm.w r4, {r4, r5, r8, r9}
        ldm.w r4, {r5, r6}
        ldm.w r5!, {r3, r8}
        ldm.w r5!, {r1, r2}
        ldm.w r2, {r1, r2}

        ldmia r4, {r4, r5, r8, r9}
        ldmia r4, {r5, r6}
        ldmia r5!, {r3, r8}
        ldm r4, {r4, r5, r8, r9}
        ldm r4, {r5, r6}
        ldm r5!, {r3, r8}
        ldmfd r5!, {r3, r8}

@ CHECK: ldm.w	r4, {r4, r5, r8, r9}    @ encoding: [0x94,0xe8,0x30,0x03]
@ CHECK: ldm.w	r4, {r5, r6}            @ encoding: [0x94,0xe8,0x60,0x00]
@ CHECK: ldm.w	r5!, {r3, r8}           @ encoding: [0xb5,0xe8,0x08,0x01]
@ CHECK: ldm.w	r4, {r4, r5, r8, r9}    @ encoding: [0x94,0xe8,0x30,0x03]
@ CHECK: ldm.w	r4, {r5, r6}            @ encoding: [0x94,0xe8,0x60,0x00]
@ CHECK: ldm.w	r5!, {r3, r8}           @ encoding: [0xb5,0xe8,0x08,0x01]
@ CHECK: ldm.w	r5!, {r1, r2}           @ encoding: [0xb5,0xe8,0x06,0x00]
@ CHECK: ldm.w	r2, {r1, r2}            @ encoding: [0x92,0xe8,0x06,0x00]

@ CHECK: ldm.w	r4, {r4, r5, r8, r9}    @ encoding: [0x94,0xe8,0x30,0x03]
@ CHECK: ldm.w	r4, {r5, r6}            @ encoding: [0x94,0xe8,0x60,0x00]
@ CHECK: ldm.w	r5!, {r3, r8}           @ encoding: [0xb5,0xe8,0x08,0x01]
@ CHECK: ldm.w	r4, {r4, r5, r8, r9}    @ encoding: [0x94,0xe8,0x30,0x03]
@ CHECK: ldm.w	r4, {r5, r6}            @ encoding: [0x94,0xe8,0x60,0x00]
@ CHECK: ldm.w	r5!, {r3, r8}           @ encoding: [0xb5,0xe8,0x08,0x01]
@ CHECK: ldm.w	r5!, {r3, r8}           @ encoding: [0xb5,0xe8,0x08,0x01]


@------------------------------------------------------------------------------
@ LDMDB
@------------------------------------------------------------------------------
        ldmdb r4, {r4, r5, r8, r9}
        ldmdb r4, {r5, r6}
        ldmdb r5!, {r3, r8}
        ldmea r5!, {r3, r8}

@ CHECK: ldmdb	r4, {r4, r5, r8, r9}    @ encoding: [0x14,0xe9,0x30,0x03]
@ CHECK: ldmdb	r4, {r5, r6}            @ encoding: [0x14,0xe9,0x60,0x00]
@ CHECK: ldmdb	r5!, {r3, r8}           @ encoding: [0x35,0xe9,0x08,0x01]
@ CHECK: ldmdb	r5!, {r3, r8}           @ encoding: [0x35,0xe9,0x08,0x01]


@------------------------------------------------------------------------------
@ LDR(immediate)
@------------------------------------------------------------------------------
        ldr r5, [r5, #-4]
        ldr r5, [r6, #32]
        ldr r5, [r6, #33]
        ldr r5, [r6, #257]
        ldr.w pc, [r7, #257]

@ CHECK: ldr	r5, [r5, #-4]           @ encoding: [0x55,0xf8,0x04,0x5c]
@ CHECK: ldr	r5, [r6, #32]           @ encoding: [0x35,0x6a]
@ CHECK: ldr.w	r5, [r6, #33]           @ encoding: [0xd6,0xf8,0x21,0x50]
@ CHECK: ldr.w	r5, [r6, #257]          @ encoding: [0xd6,0xf8,0x01,0x51]
@ CHECK: ldr.w	pc, [r7, #257]          @ encoding: [0xd7,0xf8,0x01,0xf1]


@------------------------------------------------------------------------------
@ LDR(literal)
@------------------------------------------------------------------------------
        ldr.w r5, _foo

@ CHECK: ldr.w	r5, _foo                @ encoding: [0x5f'A',0xf8'A',A,0x50'A']
            @   fixup A - offset: 0, value: _foo, kind: fixup_t2_ldst_pcrel_12


@------------------------------------------------------------------------------
@ LDR(register)
@------------------------------------------------------------------------------
        ldr r1, [r8, r1]
        ldr.w r4, [r5, r2]
        ldr r6, [r0, r2, lsl #3]
        ldr r8, [r8, r2, lsl #2]
        ldr r7, [sp, r2, lsl #1]
        ldr r7, [sp, r2, lsl #0]
        ldr r2, [r4, #255]!
        ldr r8, [sp, #4]!
        ldr lr, [sp, #-4]!
        ldr r2, [r4], #255
        ldr r8, [sp], #4
        ldr lr, [sp], #-4

@ CHECK: ldr.w	r1, [r8, r1]            @ encoding: [0x58,0xf8,0x01,0x10]
@ CHECK: ldr.w	r4, [r5, r2]            @ encoding: [0x55,0xf8,0x02,0x40]
@ CHECK: ldr.w	r6, [r0, r2, lsl #3]    @ encoding: [0x50,0xf8,0x32,0x60]
@ CHECK: ldr.w	r8, [r8, r2, lsl #2]    @ encoding: [0x58,0xf8,0x22,0x80]
@ CHECK: ldr.w	r7, [sp, r2, lsl #1]    @ encoding: [0x5d,0xf8,0x12,0x70]
@ CHECK: ldr.w	r7, [sp, r2]            @ encoding: [0x5d,0xf8,0x02,0x70]
@ CHECK: ldr	r2, [r4, #255]!         @ encoding: [0x54,0xf8,0xff,0x2f]
@ CHECK: ldr	r8, [sp, #4]!           @ encoding: [0x5d,0xf8,0x04,0x8f]
@ CHECK: ldr	lr, [sp, #-4]!          @ encoding: [0x5d,0xf8,0x04,0xed]
@ CHECK: ldr	r2, [r4], #255          @ encoding: [0x54,0xf8,0xff,0x2b]
@ CHECK: ldr	r8, [sp], #4            @ encoding: [0x5d,0xf8,0x04,0x8b]
@ CHECK: ldr	lr, [sp], #-4           @ encoding: [0x5d,0xf8,0x04,0xe9]


@------------------------------------------------------------------------------
@ LDRB(immediate)
@------------------------------------------------------------------------------
        ldrb r5, [r5, #-4]
        ldrb r5, [r6, #32]
        ldrb r5, [r6, #33]
        ldrb r5, [r6, #257]
        ldrb.w lr, [r7, #257]

@ CHECK: ldrb	r5, [r5, #-4]           @ encoding: [0x15,0xf8,0x04,0x5c]
@ CHECK: ldrb.w	r5, [r6, #32]           @ encoding: [0x96,0xf8,0x20,0x50]
@ CHECK: ldrb.w	r5, [r6, #33]           @ encoding: [0x96,0xf8,0x21,0x50]
@ CHECK: ldrb.w	r5, [r6, #257]          @ encoding: [0x96,0xf8,0x01,0x51]
@ CHECK: ldrb.w	lr, [r7, #257]          @ encoding: [0x97,0xf8,0x01,0xe1]


@------------------------------------------------------------------------------
@ LDRB(register)
@------------------------------------------------------------------------------
        ldrb r1, [r8, r1]
        ldrb.w r4, [r5, r2]
        ldrb r6, [r0, r2, lsl #3]
        ldrb r8, [r8, r2, lsl #2]
        ldrb r7, [sp, r2, lsl #1]
        ldrb r7, [sp, r2, lsl #0]

@ CHECK: ldrb.w	r1, [r8, r1]            @ encoding: [0x18,0xf8,0x01,0x10]
@ CHECK: ldrb.w	r4, [r5, r2]            @ encoding: [0x15,0xf8,0x02,0x40]
@ CHECK: ldrb.w	r6, [r0, r2, lsl #3]    @ encoding: [0x10,0xf8,0x32,0x60]
@ CHECK: ldrb.w	r8, [r8, r2, lsl #2]    @ encoding: [0x18,0xf8,0x22,0x80]
@ CHECK: ldrb.w	r7, [sp, r2, lsl #1]    @ encoding: [0x1d,0xf8,0x12,0x70]
@ CHECK: ldrb.w	r7, [sp, r2]            @ encoding: [0x1d,0xf8,0x02,0x70]


@------------------------------------------------------------------------------
@ LDRBT
@------------------------------------------------------------------------------
        ldrbt r1, [r2]
        ldrbt r1, [r8, #0]
        ldrbt r1, [r8, #3]
        ldrbt r1, [r8, #255]

@ CHECK: ldrbt	r1, [r2]                @ encoding: [0x12,0xf8,0x00,0x1e]
@ CHECK: ldrbt	r1, [r8]                @ encoding: [0x18,0xf8,0x00,0x1e]
@ CHECK: ldrbt	r1, [r8, #3]            @ encoding: [0x18,0xf8,0x03,0x1e]
@ CHECK: ldrbt	r1, [r8, #255]          @ encoding: [0x18,0xf8,0xff,0x1e]


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
