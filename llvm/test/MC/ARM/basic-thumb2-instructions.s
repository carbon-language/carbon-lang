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

        subw r11, pc, #3270
        adr.w r11, #-826

@ CHECK: subw	r11, pc, #3270          @ encoding: [0xaf,0xf6,0xc6,0x4b]
@ CHECK: adr.w	r11, #-826              @ encoding: [0xaf,0xf2,0x3a,0x3b]

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
@ CHECK: beq.w	_bar                    @ encoding: [A,0xf0'A',A,0x90'A']
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
@ BKPT
@------------------------------------------------------------------------------
        it pl
        bkpt #234

@ CHECK: it pl                      @ encoding: [0x58,0xbf]
@ CHECK: bkpt #234                    @ encoding: [0xea,0xbe]

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
        ldrb r5, [r8, #255]!
        ldrb r2, [r5, #4]!
        ldrb r1, [r4, #-4]!
        ldrb lr, [r3], #255
        ldrb r9, [r2], #4
        ldrb r3, [sp], #-4

@ CHECK: ldrb.w	r1, [r8, r1]            @ encoding: [0x18,0xf8,0x01,0x10]
@ CHECK: ldrb.w	r4, [r5, r2]            @ encoding: [0x15,0xf8,0x02,0x40]
@ CHECK: ldrb.w	r6, [r0, r2, lsl #3]    @ encoding: [0x10,0xf8,0x32,0x60]
@ CHECK: ldrb.w	r8, [r8, r2, lsl #2]    @ encoding: [0x18,0xf8,0x22,0x80]
@ CHECK: ldrb.w	r7, [sp, r2, lsl #1]    @ encoding: [0x1d,0xf8,0x12,0x70]
@ CHECK: ldrb.w	r7, [sp, r2]            @ encoding: [0x1d,0xf8,0x02,0x70]
@ CHECK: ldrb	r5, [r8, #255]!         @ encoding: [0x18,0xf8,0xff,0x5f]
@ CHECK: ldrb	r2, [r5, #4]!           @ encoding: [0x15,0xf8,0x04,0x2f]
@ CHECK: ldrb	r1, [r4, #-4]!          @ encoding: [0x14,0xf8,0x04,0x1d]
@ CHECK: ldrb	lr, [r3], #255          @ encoding: [0x13,0xf8,0xff,0xeb]
@ CHECK: ldrb	r9, [r2], #4            @ encoding: [0x12,0xf8,0x04,0x9b]
@ CHECK: ldrb	r3, [sp], #-4           @ encoding: [0x1d,0xf8,0x04,0x39]


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
@ LDRD(immediate)
@------------------------------------------------------------------------------
        ldrd r3, r5, [r6, #24]
        ldrd r3, r5, [r6, #24]!
        ldrd r3, r5, [r6], #4
        ldrd r3, r5, [r6], #-8
        ldrd r3, r5, [r6]
        ldrd r8, r1, [r3, #0]

@ CHECK: ldrd	r3, r5, [r6, #24]       @ encoding: [0xd6,0xe9,0x06,0x35]
@ CHECK: ldrd	r3, r5, [r6, #24]!      @ encoding: [0xf6,0xe9,0x06,0x35]
@ CHECK: ldrd	r3, r5, [r6], #4        @ encoding: [0xf6,0xe8,0x01,0x35]
@ CHECK: ldrd	r3, r5, [r6], #-8       @ encoding: [0x76,0xe8,0x02,0x35]
@ CHECK: ldrd	r3, r5, [r6]            @ encoding: [0xd6,0xe9,0x00,0x35]
@ CHECK: ldrd	r8, r1, [r3]            @ encoding: [0xd3,0xe9,0x00,0x81]


@------------------------------------------------------------------------------
@ FIXME: LDRD(literal)
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ LDREX/LDREXB/LDREXH/LDREXD
@------------------------------------------------------------------------------
        ldrex r1, [r4]
        ldrex r8, [r4, #0]
        ldrex r2, [sp, #128]
        ldrexb r5, [r7]
        ldrexh r9, [r12]
        ldrexd r9, r3, [r4]

@ CHECK: ldrex	r1, [r4]                @ encoding: [0x54,0xe8,0x00,0x1f]
@ CHECK: ldrex	r8, [r4]                @ encoding: [0x54,0xe8,0x00,0x8f]
@ CHECK: ldrex	r2, [sp, #128]          @ encoding: [0x5d,0xe8,0x20,0x2f]
@ CHECK: ldrexb	r5, [r7]                @ encoding: [0xd7,0xe8,0x4f,0x5f]
@ CHECK: ldrexh	r9, [r12]               @ encoding: [0xdc,0xe8,0x5f,0x9f]
@ CHECK: ldrexd	r9, r3, [r4]            @ encoding: [0xd4,0xe8,0x7f,0x93]


@------------------------------------------------------------------------------
@ LDRH(immediate)
@------------------------------------------------------------------------------
        ldrh r5, [r5, #-4]
        ldrh r5, [r6, #32]
        ldrh r5, [r6, #33]
        ldrh r5, [r6, #257]
        ldrh.w lr, [r7, #257]

@ CHECK: ldrh	r5, [r5, #-4]           @ encoding: [0x35,0xf8,0x04,0x5c]
@ CHECK: ldrh	r5, [r6, #32]           @ encoding: [0x35,0x8c]
@ CHECK: ldrh.w	r5, [r6, #33]           @ encoding: [0xb6,0xf8,0x21,0x50]
@ CHECK: ldrh.w	r5, [r6, #257]          @ encoding: [0xb6,0xf8,0x01,0x51]
@ CHECK: ldrh.w	lr, [r7, #257]          @ encoding: [0xb7,0xf8,0x01,0xe1]


@------------------------------------------------------------------------------
@ LDRH(register)
@------------------------------------------------------------------------------
        ldrh r1, [r8, r1]
        ldrh.w r4, [r5, r2]
        ldrh r6, [r0, r2, lsl #3]
        ldrh r8, [r8, r2, lsl #2]
        ldrh r7, [sp, r2, lsl #1]
        ldrh r7, [sp, r2, lsl #0]
        ldrh r5, [r8, #255]!
        ldrh r2, [r5, #4]!
        ldrh r1, [r4, #-4]!
        ldrh lr, [r3], #255
        ldrh r9, [r2], #4
        ldrh r3, [sp], #-4

@ CHECK: ldrh.w	r1, [r8, r1]            @ encoding: [0x38,0xf8,0x01,0x10]
@ CHECK: ldrh.w	r4, [r5, r2]            @ encoding: [0x35,0xf8,0x02,0x40]
@ CHECK: ldrh.w	r6, [r0, r2, lsl #3]    @ encoding: [0x30,0xf8,0x32,0x60]
@ CHECK: ldrh.w	r8, [r8, r2, lsl #2]    @ encoding: [0x38,0xf8,0x22,0x80]
@ CHECK: ldrh.w	r7, [sp, r2, lsl #1]    @ encoding: [0x3d,0xf8,0x12,0x70]
@ CHECK: ldrh.w	r7, [sp, r2]            @ encoding: [0x3d,0xf8,0x02,0x70]
@ CHECK: ldrh	r5, [r8, #255]!         @ encoding: [0x38,0xf8,0xff,0x5f]
@ CHECK: ldrh	r2, [r5, #4]!           @ encoding: [0x35,0xf8,0x04,0x2f]
@ CHECK: ldrh	r1, [r4, #-4]!          @ encoding: [0x34,0xf8,0x04,0x1d]
@ CHECK: ldrh	lr, [r3], #255          @ encoding: [0x33,0xf8,0xff,0xeb]
@ CHECK: ldrh	r9, [r2], #4            @ encoding: [0x32,0xf8,0x04,0x9b]
@ CHECK: ldrh	r3, [sp], #-4           @ encoding: [0x3d,0xf8,0x04,0x39]


@------------------------------------------------------------------------------
@ LDRH(literal)
@------------------------------------------------------------------------------
        ldrh r5, _bar

@ CHECK: ldrh.w	r5, _bar                @ encoding: [0xbf'A',0xf8'A',A,0x50'A']
@ CHECK:     @   fixup A - offset: 0, value: _bar, kind: fixup_t2_ldst_pcrel_12


@------------------------------------------------------------------------------
@ LDRSB(immediate)
@------------------------------------------------------------------------------
        ldrsb r5, [r5, #-4]
        ldrsb r5, [r6, #32]
        ldrsb r5, [r6, #33]
        ldrsb r5, [r6, #257]
        ldrsb.w lr, [r7, #257]

@ CHECK: ldrsb	r5, [r5, #-4]            @ encoding: [0x15,0xf9,0x04,0x5c]
@ CHECK: ldrsb.w r5, [r6, #32]           @ encoding: [0x96,0xf9,0x20,0x50]
@ CHECK: ldrsb.w r5, [r6, #33]           @ encoding: [0x96,0xf9,0x21,0x50]
@ CHECK: ldrsb.w r5, [r6, #257]          @ encoding: [0x96,0xf9,0x01,0x51]
@ CHECK: ldrsb.w lr, [r7, #257]          @ encoding: [0x97,0xf9,0x01,0xe1]


@------------------------------------------------------------------------------
@ LDRSB(register)
@------------------------------------------------------------------------------
        ldrsb r1, [r8, r1]
        ldrsb.w r4, [r5, r2]
        ldrsb r6, [r0, r2, lsl #3]
        ldrsb r8, [r8, r2, lsl #2]
        ldrsb r7, [sp, r2, lsl #1]
        ldrsb r7, [sp, r2, lsl #0]
        ldrsb r5, [r8, #255]!
        ldrsb r2, [r5, #4]!
        ldrsb r1, [r4, #-4]!
        ldrsb lr, [r3], #255
        ldrsb r9, [r2], #4
        ldrsb r3, [sp], #-4

@ CHECK: ldrsb.w r1, [r8, r1]           @ encoding: [0x18,0xf9,0x01,0x10]
@ CHECK: ldrsb.w r4, [r5, r2]           @ encoding: [0x15,0xf9,0x02,0x40]
@ CHECK: ldrsb.w r6, [r0, r2, lsl #3]   @ encoding: [0x10,0xf9,0x32,0x60]
@ CHECK: ldrsb.w r8, [r8, r2, lsl #2]   @ encoding: [0x18,0xf9,0x22,0x80]
@ CHECK: ldrsb.w r7, [sp, r2, lsl #1]   @ encoding: [0x1d,0xf9,0x12,0x70]
@ CHECK: ldrsb.w r7, [sp, r2]           @ encoding: [0x1d,0xf9,0x02,0x70]
@ CHECK: ldrsb	r5, [r8, #255]!         @ encoding: [0x18,0xf9,0xff,0x5f]
@ CHECK: ldrsb	r2, [r5, #4]!           @ encoding: [0x15,0xf9,0x04,0x2f]
@ CHECK: ldrsb	r1, [r4, #-4]!          @ encoding: [0x14,0xf9,0x04,0x1d]
@ CHECK: ldrsb	lr, [r3], #255          @ encoding: [0x13,0xf9,0xff,0xeb]
@ CHECK: ldrsb	r9, [r2], #4            @ encoding: [0x12,0xf9,0x04,0x9b]
@ CHECK: ldrsb	r3, [sp], #-4           @ encoding: [0x1d,0xf9,0x04,0x39]


@------------------------------------------------------------------------------
@ LDRSB(literal)
@------------------------------------------------------------------------------
        ldrsb r5, _bar

@ CHECK: ldrsb.w r5, _bar               @ encoding: [0x9f'A',0xf9'A',A,0x50'A']
@ CHECK:      @   fixup A - offset: 0, value: _bar, kind: fixup_t2_ldst_pcrel_12


@------------------------------------------------------------------------------
@ LDRSBT
@------------------------------------------------------------------------------
        ldrsbt r1, [r2]
        ldrsbt r1, [r8, #0]
        ldrsbt r1, [r8, #3]
        ldrsbt r1, [r8, #255]

@ CHECK: ldrsbt	r1, [r2]                @ encoding: [0x12,0xf9,0x00,0x1e]
@ CHECK: ldrsbt	r1, [r8]                @ encoding: [0x18,0xf9,0x00,0x1e]
@ CHECK: ldrsbt	r1, [r8, #3]            @ encoding: [0x18,0xf9,0x03,0x1e]
@ CHECK: ldrsbt	r1, [r8, #255]          @ encoding: [0x18,0xf9,0xff,0x1e]


@------------------------------------------------------------------------------
@ LDRSH(immediate)
@------------------------------------------------------------------------------
        ldrsh r5, [r5, #-4]
        ldrsh r5, [r6, #32]
        ldrsh r5, [r6, #33]
        ldrsh r5, [r6, #257]
        ldrsh.w lr, [r7, #257]

@ CHECK: ldrsh	r5, [r5, #-4]           @ encoding: [0x35,0xf9,0x04,0x5c]
@ CHECK: ldrsh.w r5, [r6, #32]          @ encoding: [0xb6,0xf9,0x20,0x50]
@ CHECK: ldrsh.w r5, [r6, #33]          @ encoding: [0xb6,0xf9,0x21,0x50]
@ CHECK: ldrsh.w r5, [r6, #257]         @ encoding: [0xb6,0xf9,0x01,0x51]
@ CHECK: ldrsh.w lr, [r7, #257]         @ encoding: [0xb7,0xf9,0x01,0xe1]


@------------------------------------------------------------------------------
@ LDRSH(register)
@------------------------------------------------------------------------------
        ldrsh r1, [r8, r1]
        ldrsh.w r4, [r5, r2]
        ldrsh r6, [r0, r2, lsl #3]
        ldrsh r8, [r8, r2, lsl #2]
        ldrsh r7, [sp, r2, lsl #1]
        ldrsh r7, [sp, r2, lsl #0]
        ldrsh r5, [r8, #255]!
        ldrsh r2, [r5, #4]!
        ldrsh r1, [r4, #-4]!
        ldrsh lr, [r3], #255
        ldrsh r9, [r2], #4
        ldrsh r3, [sp], #-4

@ CHECK: ldrsh.w r1, [r8, r1]           @ encoding: [0x38,0xf9,0x01,0x10]
@ CHECK: ldrsh.w r4, [r5, r2]           @ encoding: [0x35,0xf9,0x02,0x40]
@ CHECK: ldrsh.w r6, [r0, r2, lsl #3]   @ encoding: [0x30,0xf9,0x32,0x60]
@ CHECK: ldrsh.w r8, [r8, r2, lsl #2]   @ encoding: [0x38,0xf9,0x22,0x80]
@ CHECK: ldrsh.w r7, [sp, r2, lsl #1]   @ encoding: [0x3d,0xf9,0x12,0x70]
@ CHECK: ldrsh.w r7, [sp, r2]           @ encoding: [0x3d,0xf9,0x02,0x70]
@ CHECK: ldrsh	r5, [r8, #255]!         @ encoding: [0x38,0xf9,0xff,0x5f]
@ CHECK: ldrsh	r2, [r5, #4]!           @ encoding: [0x35,0xf9,0x04,0x2f]
@ CHECK: ldrsh	r1, [r4, #-4]!          @ encoding: [0x34,0xf9,0x04,0x1d]
@ CHECK: ldrsh	lr, [r3], #255          @ encoding: [0x33,0xf9,0xff,0xeb]
@ CHECK: ldrsh	r9, [r2], #4            @ encoding: [0x32,0xf9,0x04,0x9b]
@ CHECK: ldrsh	r3, [sp], #-4           @ encoding: [0x3d,0xf9,0x04,0x39]


@------------------------------------------------------------------------------
@ LDRSH(literal)
@------------------------------------------------------------------------------
        ldrsh r5, _bar
        ldrsh.w r4, #1435

@ CHECK: ldrsh.w r5, _bar               @ encoding: [0xbf'A',0xf9'A',A,0x50'A']
@ CHECK:      @   fixup A - offset: 0, value: _bar, kind: fixup_t2_ldst_pcrel_12
@ CHECK: ldrsh.w r4, #1435               @ encoding: [0x3f,0xf9,0x9b,0x45]

@------------------------------------------------------------------------------
@ LDRSHT
@------------------------------------------------------------------------------
        ldrsht r1, [r2]
        ldrsht r1, [r8, #0]
        ldrsht r1, [r8, #3]
        ldrsht r1, [r8, #255]

@ CHECK: ldrsht	r1, [r2]                @ encoding: [0x32,0xf9,0x00,0x1e]
@ CHECK: ldrsht	r1, [r8]                @ encoding: [0x38,0xf9,0x00,0x1e]
@ CHECK: ldrsht	r1, [r8, #3]            @ encoding: [0x38,0xf9,0x03,0x1e]
@ CHECK: ldrsht	r1, [r8, #255]          @ encoding: [0x38,0xf9,0xff,0x1e]


@------------------------------------------------------------------------------
@ LDRT
@------------------------------------------------------------------------------
        ldrt r1, [r2]
        ldrt r2, [r6, #0]
        ldrt r3, [r7, #3]
        ldrt r4, [r9, #255]

@ CHECK: ldrt	r1, [r2]                @ encoding: [0x52,0xf8,0x00,0x1e]
@ CHECK: ldrt	r2, [r6]                @ encoding: [0x56,0xf8,0x00,0x2e]
@ CHECK: ldrt	r3, [r7, #3]            @ encoding: [0x57,0xf8,0x03,0x3e]
@ CHECK: ldrt	r4, [r9, #255]          @ encoding: [0x59,0xf8,0xff,0x4e]


@------------------------------------------------------------------------------
@ LSL (immediate)
@------------------------------------------------------------------------------
        lsl r2, r3, #12
        lsls r8, r3, #31
        lsls.w r2, r3, #1
        lsl r2, r3, #4
        lsls r2, r12, #15

        lsl r3, #19
        lsls r8, #2
        lsls.w r7, #5
        lsl.w r12, #21

@ CHECK: lsl.w	r2, r3, #12             @ encoding: [0x4f,0xea,0x03,0x32]
@ CHECK: lsls.w	r8, r3, #31             @ encoding: [0x5f,0xea,0xc3,0x78]
@ CHECK: lsls.w	r2, r3, #1              @ encoding: [0x5f,0xea,0x43,0x02]
@ CHECK: lsl.w	r2, r3, #4              @ encoding: [0x4f,0xea,0x03,0x12]
@ CHECK: lsls.w	r2, r12, #15            @ encoding: [0x5f,0xea,0xcc,0x32]

@ CHECK: lsl.w	r3, r3, #19             @ encoding: [0x4f,0xea,0xc3,0x43]
@ CHECK: lsls.w	r8, r8, #2              @ encoding: [0x5f,0xea,0x88,0x08]
@ CHECK: lsls.w	r7, r7, #5              @ encoding: [0x5f,0xea,0x47,0x17]
@ CHECK: lsl.w	r12, r12, #21           @ encoding: [0x4f,0xea,0x4c,0x5c]


@------------------------------------------------------------------------------
@ LSL (register)
@------------------------------------------------------------------------------
        lsl r3, r4, r2
        lsl.w r1, r2
        lsls r3, r4, r8

@ CHECK: lsl.w	r3, r4, r2              @ encoding: [0x04,0xfa,0x02,0xf3]
@ CHECK: lsl.w	r1, r1, r2              @ encoding: [0x01,0xfa,0x02,0xf1]
@ CHECK: lsls.w	r3, r4, r8              @ encoding: [0x14,0xfa,0x08,0xf3]


@------------------------------------------------------------------------------
@ LSR (immediate)
@------------------------------------------------------------------------------
        lsr r2, r3, #12
        lsrs r8, r3, #32
        lsrs.w r2, r3, #1
        lsr r2, r3, #4
        lsrs r2, r12, #15

        lsr r3, #19
        lsrs r8, #2
        lsrs.w r7, #5
        lsr.w r12, #21

@ CHECK: lsr.w	r2, r3, #12             @ encoding: [0x4f,0xea,0x13,0x32]
@ CHECK: lsrs.w	r8, r3, #32             @ encoding: [0x5f,0xea,0x13,0x08]
@ CHECK: lsrs.w	r2, r3, #1              @ encoding: [0x5f,0xea,0x53,0x02]
@ CHECK: lsr.w	r2, r3, #4              @ encoding: [0x4f,0xea,0x13,0x12]
@ CHECK: lsrs.w	r2, r12, #15            @ encoding: [0x5f,0xea,0xdc,0x32]

@ CHECK: lsr.w	r3, r3, #19             @ encoding: [0x4f,0xea,0xd3,0x43]
@ CHECK: lsrs.w	r8, r8, #2              @ encoding: [0x5f,0xea,0x98,0x08]
@ CHECK: lsrs.w	r7, r7, #5              @ encoding: [0x5f,0xea,0x57,0x17]
@ CHECK: lsr.w	r12, r12, #21           @ encoding: [0x4f,0xea,0x5c,0x5c]


@------------------------------------------------------------------------------
@ LSR (register)
@------------------------------------------------------------------------------
        lsr r3, r4, r2
        lsr.w r1, r2
        lsrs r3, r4, r8

@ CHECK: lsr.w	r3, r4, r2              @ encoding: [0x24,0xfa,0x02,0xf3]
@ CHECK: lsr.w	r1, r1, r2              @ encoding: [0x21,0xfa,0x02,0xf1]
@ CHECK: lsrs.w	r3, r4, r8              @ encoding: [0x34,0xfa,0x08,0xf3]

@------------------------------------------------------------------------------
@ MCR/MCR2
@------------------------------------------------------------------------------
        mcr  p7, #1, r5, c1, c1, #4
        mcr2  p7, #1, r5, c1, c1, #4

@ CHECK: mcr	p7, #1, r5, c1, c1, #4  @ encoding: [0x21,0xee,0x91,0x57]
@ CHECK: mcr2	p7, #1, r5, c1, c1, #4  @ encoding: [0x21,0xfe,0x91,0x57]


@------------------------------------------------------------------------------
@ MCRR/MCRR2
@------------------------------------------------------------------------------
        mcrr  p7, #15, r5, r4, c1
        mcrr2  p7, #15, r5, r4, c1

@ CHECK: mcrr	p7, #15, r5, r4, c1     @ encoding: [0x44,0xec,0xf1,0x57]
@ CHECK: mcrr2	p7, #15, r5, r4, c1     @ encoding: [0x44,0xfc,0xf1,0x57]


@------------------------------------------------------------------------------
@ MLA/MLS
@------------------------------------------------------------------------------
        mla  r1,r2,r3,r4
        mls  r1,r2,r3,r4

@ CHECK: mla	r1, r2, r3, r4          @ encoding: [0x02,0xfb,0x03,0x41]
@ CHECK: mls	r1, r2, r3, r4          @ encoding: [0x02,0xfb,0x13,0x41]


@------------------------------------------------------------------------------
@ MOV(immediate)
@------------------------------------------------------------------------------
        movs r1, #21
        movs.w r1, #21
        movs r8, #21
        movw r0, #65535
        movw r1, #43777
        movw r1, #43792
        mov.w r0, #0x3fc0000
        mov r0, #0x3fc0000
        movs.w r0, #0x3fc0000
        itte eq
        movseq r1, #12
        moveq r1, #12
        movne.w r1, #12

@ CHECK: movs	r1, #21                 @ encoding: [0x15,0x21]
@ CHECK: movs.w	r1, #21                 @ encoding: [0x5f,0xf0,0x15,0x01]
@ CHECK: movs.w	r8, #21                 @ encoding: [0x5f,0xf0,0x15,0x08]
@ CHECK: movw	r0, #65535              @ encoding: [0x4f,0xf6,0xff,0x70]
@ CHECK: movw	r1, #43777              @ encoding: [0x4a,0xf6,0x01,0x31]
@ CHECK: movw	r1, #43792              @ encoding: [0x4a,0xf6,0x10,0x31]
@ CHECK: mov.w	r0, #66846720           @ encoding: [0x4f,0xf0,0x7f,0x70]
@ CHECK: mov.w	r0, #66846720           @ encoding: [0x4f,0xf0,0x7f,0x70]
@ CHECK: movs.w	r0, #66846720           @ encoding: [0x5f,0xf0,0x7f,0x70]
@ CHECK: itte	eq                      @ encoding: [0x06,0xbf]
@ CHECK: movseq.w	r1, #12         @ encoding: [0x5f,0xf0,0x0c,0x01]
@ CHECK: moveq	r1, #12                 @ encoding: [0x0c,0x21]
@ CHECK: movne.w r1, #12                @ encoding: [0x4f,0xf0,0x0c,0x01]


@------------------------------------------------------------------------------
@ MOVT
@------------------------------------------------------------------------------
        movt r3, #7
        movt r6, #0xffff
        it eq
        movteq r4, #0xff0

@ CHECK: movt	r3, #7                  @ encoding: [0xc0,0xf2,0x07,0x03]
@ CHECK: movt	r6, #65535              @ encoding: [0xcf,0xf6,0xff,0x76]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: movteq	r4, #4080               @ encoding: [0xc0,0xf6,0xf0,0x74]

@------------------------------------------------------------------------------
@ MRC/MRC2
@------------------------------------------------------------------------------
        mrc  p14, #0, r1, c1, c2, #4
        mrc2  p14, #0, r1, c1, c2, #4

@ CHECK: mrc	p14, #0, r1, c1, c2, #4 @ encoding: [0x11,0xee,0x92,0x1e]
@ CHECK: mrc2	p14, #0, r1, c1, c2, #4 @ encoding: [0x11,0xfe,0x92,0x1e]


@------------------------------------------------------------------------------
@ MRRC/MRRC2
@------------------------------------------------------------------------------
        mrrc  p7, #1, r5, r4, c1
        mrrc2  p7, #1, r5, r4, c1

@ CHECK: mrrc	p7, #1, r5, r4, c1      @ encoding: [0x54,0xec,0x11,0x57]
@ CHECK: mrrc2	p7, #1, r5, r4, c1      @ encoding: [0x54,0xfc,0x11,0x57]


@------------------------------------------------------------------------------
@ MRS
@------------------------------------------------------------------------------
        mrs  r8, apsr
        mrs  r8, cpsr
        mrs  r8, spsr

@ CHECK: mrs	r8, apsr                @ encoding: [0xef,0xf3,0x00,0x88]
@ CHECK: mrs	r8, apsr                @ encoding: [0xef,0xf3,0x00,0x88]
@ CHECK: mrs	r8, spsr                @ encoding: [0xff,0xf3,0x00,0x88]


@------------------------------------------------------------------------------
@ MSR
@------------------------------------------------------------------------------
        msr  apsr, r1
        msr  apsr_g, r2
        msr  apsr_nzcvq, r3
        msr  APSR_nzcvq, r4
        msr  apsr_nzcvqg, r5
        msr  cpsr_fc, r6
        msr  cpsr_c, r7
        msr  cpsr_x, r8
        msr  cpsr_fc, r9
        msr  cpsr_all, r11
        msr  cpsr_fsx, r12
        msr  spsr_fc, r0
        msr  SPSR_fsxc, r5
        msr  cpsr_fsxc, r8

@ CHECK: msr	APSR_nzcvq, r1          @ encoding: [0x81,0xf3,0x00,0x88]
@ CHECK: msr	APSR_g, r2              @ encoding: [0x82,0xf3,0x00,0x84]
@ CHECK: msr	APSR_nzcvq, r3          @ encoding: [0x83,0xf3,0x00,0x88]
@ CHECK: msr	APSR_nzcvq, r4          @ encoding: [0x84,0xf3,0x00,0x88]
@ CHECK: msr	APSR_nzcvqg, r5         @ encoding: [0x85,0xf3,0x00,0x8c]
@ CHECK: msr	CPSR_fc, r6             @ encoding: [0x86,0xf3,0x00,0x89]
@ CHECK: msr	CPSR_c, r7              @ encoding: [0x87,0xf3,0x00,0x81]
@ CHECK: msr	CPSR_x, r8              @ encoding: [0x88,0xf3,0x00,0x82]
@ CHECK: msr	CPSR_fc, r9             @ encoding: [0x89,0xf3,0x00,0x89]
@ CHECK: msr	CPSR_fc, r11            @ encoding: [0x8b,0xf3,0x00,0x89]
@ CHECK: msr	CPSR_fsx, r12           @ encoding: [0x8c,0xf3,0x00,0x8e]
@ CHECK: msr	SPSR_fc, r0             @ encoding: [0x90,0xf3,0x00,0x89]
@ CHECK: msr	SPSR_fsxc, r5           @ encoding: [0x95,0xf3,0x00,0x8f]
@ CHECK: msr	CPSR_fsxc, r8           @ encoding: [0x88,0xf3,0x00,0x8f]


@------------------------------------------------------------------------------
@ MUL
@------------------------------------------------------------------------------
        muls r3, r4, r3
        mul r3, r4, r3
        mul r3, r4, r6
        it eq
        muleq r3, r4, r5

@ CHECK: muls	r3, r4, r3              @ encoding: [0x63,0x43]
@ CHECK: mul	r3, r4, r3              @ encoding: [0x04,0xfb,0x03,0xf3]
@ CHECK: mul	r3, r4, r6              @ encoding: [0x04,0xfb,0x06,0xf3]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: muleq	r3, r4, r5              @ encoding: [0x04,0xfb,0x05,0xf3]


@------------------------------------------------------------------------------
@ MVN(immediate)
@------------------------------------------------------------------------------
        mvns r8, #21
        mvn r0, #0x3fc0000
        mvns r0, #0x3fc0000
        itte eq
        mvnseq r1, #12
        mvneq r1, #12
        mvnne r1, #12

@ CHECK: mvns	r8, #21                 @ encoding: [0x7f,0xf0,0x15,0x08]
@ CHECK: mvn	r0, #66846720           @ encoding: [0x6f,0xf0,0x7f,0x70]
@ CHECK: mvns	r0, #66846720           @ encoding: [0x7f,0xf0,0x7f,0x70]
@ CHECK: itte	eq                      @ encoding: [0x06,0xbf]
@ CHECK: mvnseq	r1, #12                 @ encoding: [0x7f,0xf0,0x0c,0x01]
@ CHECK: mvneq	r1, #12                 @ encoding: [0x6f,0xf0,0x0c,0x01]
@ CHECK: mvnne	r1, #12                 @ encoding: [0x6f,0xf0,0x0c,0x01]


@------------------------------------------------------------------------------
@ MVN(register)
@------------------------------------------------------------------------------
        mvn r2, r3
        mvns r2, r3
        mvn r5, r6, lsl #19
        mvn r5, r6, lsr #9
        mvn r5, r6, asr #4
        mvn r5, r6, ror #6
        mvn r5, r6, rrx
        it eq
        mvneq r2, r3

@ CHECK: mvn.w	r2, r3                  @ encoding: [0x6f,0xea,0x03,0x02]
@ CHECK: mvns	r2, r3                  @ encoding: [0xda,0x43]
@ CHECK: mvn.w	r5, r6, lsl #19         @ encoding: [0x6f,0xea,0xc6,0x45]
@ CHECK: mvn.w	r5, r6, lsr #9          @ encoding: [0x6f,0xea,0x56,0x25]
@ CHECK: mvn.w	r5, r6, asr #4          @ encoding: [0x6f,0xea,0x26,0x15]
@ CHECK: mvn.w	r5, r6, ror #6          @ encoding: [0x6f,0xea,0xb6,0x15]
@ CHECK: mvn.w	r5, r6, rrx             @ encoding: [0x6f,0xea,0x36,0x05]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: mvneq	r2, r3                  @ encoding: [0xda,0x43]

@------------------------------------------------------------------------------
@ NOP
@------------------------------------------------------------------------------
        nop.w

@ CHECK: nop.w                          @ encoding: [0xaf,0xf3,0x00,0x80]


@------------------------------------------------------------------------------
@ ORN
@------------------------------------------------------------------------------
        orn r4, r5, #0xf000
        orn r4, r5, r6
        orns r4, r5, r6
        orn r4, r5, r6, lsl #5
        orns r4, r5, r6, lsr #5
        orn r4, r5, r6, lsr #5
        orns r4, r5, r6, asr #5
        orn r4, r5, r6, ror #5

@ CHECK: orn	r4, r5, #61440          @ encoding: [0x65,0xf4,0x70,0x44]
@ CHECK: orn	r4, r5, r6              @ encoding: [0x65,0xea,0x06,0x04]
@ CHECK: orns	r4, r5, r6              @ encoding: [0x75,0xea,0x06,0x04]
@ CHECK: orn	r4, r5, r6, lsl #5      @ encoding: [0x65,0xea,0x46,0x14]
@ CHECK: orns	r4, r5, r6, lsr #5      @ encoding: [0x75,0xea,0x56,0x14]
@ CHECK: orn	r4, r5, r6, lsr #5      @ encoding: [0x65,0xea,0x56,0x14]
@ CHECK: orns	r4, r5, r6, asr #5      @ encoding: [0x75,0xea,0x66,0x14]
@ CHECK: orn	r4, r5, r6, ror #5      @ encoding: [0x65,0xea,0x76,0x14]


@------------------------------------------------------------------------------
@ ORR
@------------------------------------------------------------------------------
        orr r4, r5, #0xf000
        orr r4, r5, r6
        orr r4, r5, r6, lsl #5
        orrs r4, r5, r6, lsr #5
        orr r4, r5, r6, lsr #5
        orrs r4, r5, r6, asr #5
        orr r4, r5, r6, ror #5

@ CHECK: orr	r4, r5, #61440          @ encoding: [0x45,0xf4,0x70,0x44]
@ CHECK: orr.w	r4, r5, r6              @ encoding: [0x45,0xea,0x06,0x04]
@ CHECK: orr.w	r4, r5, r6, lsl #5      @ encoding: [0x45,0xea,0x46,0x14]
@ CHECK: orrs.w	r4, r5, r6, lsr #5      @ encoding: [0x55,0xea,0x56,0x14]
@ CHECK: orr.w	r4, r5, r6, lsr #5      @ encoding: [0x45,0xea,0x56,0x14]
@ CHECK: orrs.w	r4, r5, r6, asr #5      @ encoding: [0x55,0xea,0x66,0x14]
@ CHECK: orr.w	r4, r5, r6, ror #5      @ encoding: [0x45,0xea,0x76,0x14]


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

@------------------------------------------------------------------------------
@ SUB (register)
@------------------------------------------------------------------------------
        sub.w r5, r2, r12, rrx

@ CHECK: sub.w r5, r2, r12, rrx        @ encoding: [0xa2,0xeb,0x3c,0x05]

