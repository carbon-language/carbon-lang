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

        ! CHECK: ba .BB0      ! encoding: [0x10,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        ba .BB0

        ! CHECK: bne .BB0     ! encoding: [0x12,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bne .BB0

        ! CHECK: be .BB0      ! encoding: [0x02,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        be .BB0

        ! CHECK: bg .BB0      ! encoding: [0x14,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bg .BB0

        ! CHECK: ble .BB0      ! encoding: [0x04,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        ble .BB0

        ! CHECK: bge .BB0      ! encoding: [0x16,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bge .BB0

        ! CHECK: bl .BB0      ! encoding: [0x06,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bl .BB0

        ! CHECK: bgu .BB0      ! encoding: [0x18,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bgu .BB0

        ! CHECK: bleu .BB0      ! encoding: [0x08,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bleu .BB0

        ! CHECK: bcc .BB0      ! encoding: [0x1a,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bcc .BB0

        ! CHECK: bcs .BB0      ! encoding: [0x0a,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bcs .BB0

        ! CHECK: bpos .BB0      ! encoding: [0x1c,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bpos .BB0

        ! CHECK: bneg .BB0      ! encoding: [0x0c,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bneg .BB0

        ! CHECK: bvc .BB0      ! encoding: [0x1e,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bvc .BB0

        ! CHECK: bvs .BB0      ! encoding: [0x0e,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bvs .BB0

        ! CHECK:             fbu .BB0                        ! encoding: [0x0f,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbu .BB0

        ! CHECK:             fbg .BB0                        ! encoding: [0x0d,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbg .BB0
        ! CHECK:             fbug .BB0                       ! encoding: [0x0b,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbug .BB0

        ! CHECK:             fbl .BB0                        ! encoding: [0x09,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbl .BB0

        ! CHECK:             fbul .BB0                       ! encoding: [0x07,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbul .BB0

        ! CHECK:             fblg .BB0                       ! encoding: [0x05,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fblg .BB0

        ! CHECK:             fbne .BB0                       ! encoding: [0x03,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbne .BB0

        ! CHECK:             fbe .BB0                        ! encoding: [0x13,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbe .BB0

        ! CHECK:             fbue .BB0                       ! encoding: [0x15,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbue .BB0

        ! CHECK:             fbge .BB0                       ! encoding: [0x17,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbge .BB0

        ! CHECK:             fbuge .BB0                      ! encoding: [0x19,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbuge .BB0

        ! CHECK:             fble .BB0                       ! encoding: [0x1b,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fble .BB0

        ! CHECK:             fbule .BB0                      ! encoding: [0x1d,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbule .BB0

        ! CHECK:             fbo .BB0                        ! encoding: [0x1f,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbo .BB0
