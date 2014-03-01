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

