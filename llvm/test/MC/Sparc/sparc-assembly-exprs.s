! RUN: llvm-mc %s -arch=sparc   -show-encoding | FileCheck %s

        ! CHECK: mov 1033, %o1  ! encoding: [0x92,0x10,0x24,0x09]
        mov      (0x400|9), %o1
        ! CHECK: mov 60, %o2    ! encoding: [0x94,0x10,0x20,0x3c]
        mov      (12+3<<2), %o2
