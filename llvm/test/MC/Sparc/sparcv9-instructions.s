! RUN: not llvm-mc %s -arch=sparc   -show-encoding 2>&1 | FileCheck %s --check-prefix=V8
! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s --check-prefix=V9

        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: addc %g2, %g1, %g3
        ! V9:      addx %g2, %g1, %g3              ! encoding: [0x86,0x40,0x80,0x01]
        addc %g2, %g1, %g3

        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: addccc %g1, %g2, %g3
        ! V9:      addxcc %g1, %g2, %g3            ! encoding: [0x86,0xc0,0x40,0x02]
        addccc %g1, %g2, %g3

        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: subc %g2, %g1, %g3
        ! V9:      subx %g2, %g1, %g3          ! encoding: [0x86,0x60,0x80,0x01]
        subc %g2, %g1, %g3

        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: subccc %g1, %g2, %g3
        ! V9:      subxcc %g1, %g2, %g3         ! encoding: [0x86,0xe0,0x40,0x02]
        subccc %g1, %g2, %g3

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: popc %g1, %g2
        ! V9:      popc %g1, %g2                ! encoding: [0x85,0x70,0x00,0x01]
        popc %g1, %g2


        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: signx %g1, %g2
        ! V9: sra %g1, %g0, %g2               ! encoding: [0x85,0x38,0x40,0x00]
        signx %g1, %g2
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: signx %g1
        ! V9: sra %g1, %g0, %g1               ! encoding: [0x83,0x38,0x40,0x00]
        signx %g1

        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: lduw [%i0 + %l6], %o2
        ! V9: ld [%i0+%l6], %o2    ! encoding: [0xd4,0x06,0x00,0x16]
        lduw [%i0 + %l6], %o2
        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: lduw [%i0 + 32], %o2
        ! V9: ld [%i0+32], %o2     ! encoding: [0xd4,0x06,0x20,0x20]
        lduw [%i0 + 32], %o2
        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: lduw [%g1], %o2
        ! V9: ld [%g1], %o2        ! encoding: [0xd4,0x00,0x40,0x00]
        lduw [%g1], %o2
        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: lduwa [%i0 + %l6] 131, %o2
        ! V9: lda [%i0+%l6] 131, %o2 ! encoding: [0xd4,0x86,0x10,0x76]
        lduwa [%i0 + %l6] 131, %o2
