@ RUN: not llvm-mc -triple=thumbv7-apple-darwin < %s 2> %t
@ RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

@ Ill-formed IT block instructions.
        itet eq
        addle r0, r1, r2
        nop
        it le
        iteeee gt
        ittfe le
        nopeq

@ CHECK-ERRORS: error: incorrect condition in IT block; got 'le', but expected 'eq'
@ CHECK-ERRORS:         addle r0, r1, r2
@ CHECK-ERRORS:            ^
@ CHECK-ERRORS: error: incorrect condition in IT block; got 'al', but expected 'ne'
@ CHECK-ERRORS:         nop
@ CHECK-ERRORS:            ^
@ CHECK-ERRORS: error: instructions in IT block must be predicable
@ CHECK-ERRORS:         it le
@ CHECK-ERRORS:         ^
@ CHECK-ERRORS: error: too many conditions on IT instruction
@ CHECK-ERRORS:         iteeee gt
@ CHECK-ERRORS:           ^
@ CHECK-ERRORS: error: illegal IT block condition mask 'tfe'
@ CHECK-ERRORS:         ittfe le
@ CHECK-ERRORS:           ^
@ CHECK-ERRORS: error: predicated instructions must be in IT block
@ CHECK-ERRORS:         nopeq
@ CHECK-ERRORS:         ^
