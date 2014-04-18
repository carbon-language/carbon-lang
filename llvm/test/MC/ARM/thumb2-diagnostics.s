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

        @ Out of range immediates for MRC/MRC2/MRRC/MRRC2
        mrc  p14, #8, r1, c1, c2, #4
        mrc  p14, #1, r1, c1, c2, #8
        mrc2  p14, #8, r1, c1, c2, #4
        mrc2  p14, #0, r1, c1, c2, #9
        mrrc  p7, #16, r5, r4, c1
        mrrc2  p7, #17, r5, r4, c1
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: invalid operand for instruction
@ CHECK-ERRORS: error: immediate operand must be in the range [0,15]
@ CHECK-ERRORS: error: immediate operand must be in the range [0,15]

        isb  #-1
        isb  #16
@ CHECK-ERRORS: error: immediate value out of range
@ CHECK-ERRORS: error: immediate value out of range

        itt eq
        bkpteq #1
@ CHECK-ERRORS: error: instruction 'bkpt' is not predicable, but condition code specified

        nopeq
        nopeq

@ out of range operands for Thumb2 targets

        beq.w  #-1048578
        bne.w  #1048576
        blt.w  #1013411
        b.w    #-16777218
        b.w    #16777216
        b.w    #1592313

@ CHECK-ERRORS: error: branch target out of range
@ CHECK-ERRORS: error: branch target out of range
@ CHECK-ERRORS: error: branch target out of range
@ CHECK-ERRORS: error: branch target out of range
@ CHECK-ERRORS: error: branch target out of range
@ CHECK-ERRORS: error: branch target out of range

foo2:
        mov r0, foo2
        movw r0, foo2
        movt r0, foo2
@ CHECK-ERRORS: error: immediate expression for mov requires :lower16: or :upper16
@ CHECK-ERRORS:                 ^
@ CHECK-ERRORS: error: immediate expression for mov requires :lower16: or :upper16
@ CHECK-ERRORS:                  ^
@ CHECK-ERRORS: error: immediate expression for mov requires :lower16: or :upper16
@ CHECK-ERRORS:                  ^
