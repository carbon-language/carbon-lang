@ RUN: not llvm-mc -triple=armv7-apple-darwin < %s 2> %t
@ RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

@ Check for various assembly diagnostic messages on invalid input.

@ 's' bit on an instruction that can't accept it.
        mlss r1, r2, r3, r4
@ CHECK-ERRORS: error: instruction 'mls' can not set flags,
@ CHECK-ERRORS: but 's' suffix specified


        @ Out of range shift immediate values.
        adc r1, r2, r3, lsl #invalid
        adc r4, r5, r6, lsl #-1
        adc r4, r5, r6, lsl #32
        adc r4, r5, r6, lsr #-1
        adc r4, r5, r6, lsr #33
        adc r4, r5, r6, asr #-1
        adc r4, r5, r6, asr #33
        adc r4, r5, r6, ror #-1
        adc r4, r5, r6, ror #32

@ CHECK-ERRORS: error: invalid immediate shift value
@ CHECK-ERRORS:         adc r1, r2, r3, lsl #invalid
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, lsl #-1
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, lsl #32
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, lsr #-1
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, lsr #33
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, asr #-1
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, asr #33
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, ror #-1
@ CHECK-ERRORS:                              ^
@ CHECK-ERRORS: error: immediate shift value out of range
@ CHECK-ERRORS:         adc r4, r5, r6, ror #32


        @ Out of range 16-bit immediate on BKPT
        bkpt #65536

@ CHECK-ERRORS: error: invalid operand for instruction

        @ Out of range 4 and 3 bit immediates on CDP[2]

        @ Out of range immediates for CDP/CDP2
        cdp  p7, #2, c1, c1, c1, #8
        cdp  p7, #1, c1, c1, c1, #8
        cdp2  p7, #2, c1, c1, c1, #8
        cdp2  p7, #1, c1, c1, c1, #8

@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction

        @ Out of range immediates for DBG
        dbg #-1
        dbg #16

@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@  Double-check that we're synced up with the right diagnostics.
@ CHECK-ERRORS: dbg #16

        @ Out of range immediate for MCR/MCR2/MCRR/MCRR2
        mcr  p7, #8, r5, c1, c1, #4
        mcr  p7, #2, r5, c1, c1, #8
        mcr2  p7, #8, r5, c1, c1, #4
        mcr2  p7, #1, r5, c1, c1, #8
        mcrr  p7, #16, r5, r4, c1
        mcrr2  p7, #16, r5, r4, c1
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction


        @ Out of range immediate for MOV
        movw r9, 0x10000
@ CHECK-ERRORS: error: invalid operand for instruction
