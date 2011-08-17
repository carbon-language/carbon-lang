@ RUN: not llvm-mc -triple=thumbv6-apple-darwin < %s 2> %t
@ RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

@ Check for various assembly diagnostic messages on invalid input.

@ ADD instruction w/o 'S' suffix.
        add r1, r2, r3
@ CHECK-ERRORS: error: invalid instruction
@ CHECK-ERRORS:         add r1, r2, r3
@ CHECK-ERRORS:         ^

@ Instructions which require v6+ for both registers to be low regs.
        add r2, r3
        mov r2, r3
@ CHECK-ERRORS: error: instruction variant requires Thumb2
@ CHECK-ERRORS:         add r2, r3
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: instruction variant requires ARMv6 or later
@ CHECK-ERRORS:         mov r2, r3
@ CHECK-ERRORS:         ^


@ Out of range immediates for ASR instruction.
        asrs r2, r3, #33
        asrs r2, r3, #0
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         asrs r2, r3, #33
@ CHECK-ERRORS:                      ^
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS:         asrs r2, r3, #0
@ CHECK-ERRORS:                      ^
