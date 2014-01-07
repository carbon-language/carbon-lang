! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s

        ! CHECK: call foo     ! encoding: [0b01AAAAAA,A,A,A]
        ! CHECK:              !   fixup A - offset: 0, value: foo, kind: fixup_sparc_call30
        call foo

        ! CHECK: or %g1, %lo(sym), %g3 ! encoding: [0x86,0x10,0b011000AA,A]
        ! CHECK-NEXT                   !   fixup A - offset: 0, value: %lo(sym), kind: fixup_sparc_lo10
        or %g1, %lo(sym), %g3

        ! CHECK: sethi %hi(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        ! CHECK-NEXT:                 !   fixup A - offset: 0, value: %hi(sym), kind: fixup_sparc_hi22
        sethi %hi(sym), %l0

        ! CHECK: sethi %h44(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        ! CHECK-NEXT:                 !   fixup A - offset: 0, value: %h44(sym), kind: fixup_sparc_h44
        sethi %h44(sym), %l0

        ! CHECK: or %g1, %m44(sym), %g3 ! encoding: [0x86,0x10,0b011000AA,A]
        ! CHECK-NEXT                   !   fixup A - offset: 0, value: %m44(sym), kind: fixup_sparc_m44
        or %g1, %m44(sym), %g3

        ! CHECK: or %g1, %l44(sym), %g3 ! encoding: [0x86,0x10,0b0110AAAA,A]
        ! CHECK-NEXT                   !   fixup A - offset: 0, value: %l44(sym), kind: fixup_sparc_l44
        or %g1, %l44(sym), %g3

        ! CHECK: sethi %hh(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        ! CHECK-NEXT:                 !   fixup A - offset: 0, value: %hh(sym), kind: fixup_sparc_hh
        sethi %hh(sym), %l0

        ! CHECK: or %g1, %hm(sym), %g3 ! encoding: [0x86,0x10,0b011000AA,A]
        ! CHECK-NEXT                   !   fixup A - offset: 0, value: %hm(sym), kind: fixup_sparc_hm
        or %g1, %hm(sym), %g3
