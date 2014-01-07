! RUN: llvm-mc %s -arch=sparc   -show-encoding | FileCheck %s
! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s

        ! CHECK: call foo
        call foo

        ! CHECK: call %g1+%i2
        call %g1 + %i2

        ! CHECK: call %o1+8
        call %o1 + 8

        ! CHECK: call %g1
        call %g1

        ! CHECK: jmp %g1+%i2
        jmp %g1 + %i2

        ! CHECK: jmp %o1+8
        jmp %o1 + 8

        ! CHECK: jmp %g1
        jmp %g1
