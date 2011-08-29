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
