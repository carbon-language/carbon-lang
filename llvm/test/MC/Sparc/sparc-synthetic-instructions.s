! RUN: llvm-mc %s -arch=sparc   -show-encoding | FileCheck %s
! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s

! Section A.3 Synthetic Instructions
        ! CHECK: sethi %hi(40000), %g1            ! encoding: [0x03,0b00AAAAAA,A,A]
        ! CHECK:                                  !   fixup A - offset: 0, value: %hi(40000), kind: fixup_sparc_hi22
        ! CHECK: or %g1, %lo(40000), %g1          ! encoding: [0x82,0x10,0b011000AA,A]
        ! CHECK:                                  !   fixup A - offset: 0, value: %lo(40000), kind: fixup_sparc_lo10
        set 40000, %g1
        ! CHECK: mov      %lo(1), %g1             ! encoding: [0x82,0x10,0b001000AA,A]
        ! CHECK:                                  !   fixup A - offset: 0, value: %lo(1), kind: fixup_sparc_lo10
        set 1, %g1

        ! CHECK: sethi %hi(32768), %g1            ! encoding: [0x03,0b00AAAAAA,A,A]
        ! CHECK:                                  !   fixup A - offset: 0, value: %hi(32768), kind: fixup_sparc_hi22
        set 32768, %g1

