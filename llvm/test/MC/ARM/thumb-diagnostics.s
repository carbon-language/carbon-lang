@ RUN: not llvm-mc -triple=thumbv6-apple-darwin < %s 2> %t
@ RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

@ Check for various assembly diagnostic messages on invalid input.

@ ADD instruction w/o 'S' suffix.
        add r1, r2, r3
@ CHECK-ERRORS: error: invalid instruction
@ CHECK-ERRORS:         add r1, r2, r3
@ CHECK-ERRORS:         ^
