@ RUN: not llvm-mc -triple=armv7-apple-darwin < %s 2> %t
@ RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

@ Check for various assembly diagnostic messages on invalid input.

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
