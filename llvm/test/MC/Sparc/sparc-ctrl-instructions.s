! RUN: llvm-mc %s -arch=sparc   -show-encoding | FileCheck %s
! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s

        ! CHECK: call foo     ! encoding: [0b01AAAAAA,A,A,A]
        ! CHECK:              !   fixup A - offset: 0, value: foo, kind: fixup_sparc_call30
        call foo

        ! CHECK: call %g1+%i2 ! encoding: [0x9f,0xc0,0x40,0x1a]
        call %g1 + %i2

        ! CHECK: call %o1+8   ! encoding: [0x9f,0xc2,0x60,0x08]
        call %o1 + 8

        ! CHECK: call %g1     ! encoding: [0x9f,0xc0,0x60,0x00]
        call %g1

        ! CHECK: call %g1+%lo(sym)   ! encoding: [0x9f,0xc0,0b011000AA,A]
        ! CHECK-NEXT:                ! fixup A - offset: 0, value: %lo(sym), kind: fixup_sparc_lo10
        call %g1+%lo(sym)

        ! CHECK: jmp %g1+%i2  ! encoding: [0x81,0xc0,0x40,0x1a]
        jmp %g1 + %i2

        ! CHECK: jmp %o1+8    ! encoding: [0x81,0xc2,0x60,0x08]
        jmp %o1 + 8

        ! CHECK: jmp %g1      ! encoding: [0x81,0xc0,0x60,0x00]
        jmp %g1

        ! CHECK: jmp %g1+%lo(sym)   ! encoding: [0x81,0xc0,0b011000AA,A]
        ! CHECK-NEXT:                ! fixup A - offset: 0, value: %lo(sym), kind: fixup_sparc_lo10
        jmp %g1+%lo(sym)

