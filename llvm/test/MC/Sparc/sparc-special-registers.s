! RUN: llvm-mc %s -arch=sparc -show-encoding | FileCheck %s
! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s

        ! CHECK: rd %y, %i0            ! encoding: [0xb1,0x40,0x00,0x00]
        rd %y, %i0

        ! CHECK: rd %asr1, %i0         ! encoding: [0xb1,0x40,0x40,0x00]
        rd %asr1, %i0

        ! CHECK: wr %i0, 5, %y         ! encoding: [0x81,0x86,0x20,0x05]
        wr %i0, 5, %y

        ! CHECK: wr %i0, %i1, %asr15   ! encoding: [0x9f,0x86,0x00,0x19]
        wr %i0, %i1, %asr15

        ! CHECK: rd %asr15, %g0        ! encoding: [0x81,0x43,0xc0,0x00]
        rd %asr15, %g0

        ! CHECK: rd %psr, %i0          ! encoding: [0xb1,0x48,0x00,0x00]
        rd %psr, %i0

        ! CHECK: rd %wim, %i0          ! encoding: [0xb1,0x50,0x00,0x00]
        rd %wim, %i0

        ! CHECK: rd %tbr, %i0          ! encoding: [0xb1,0x58,0x00,0x00]
        rd %tbr, %i0

        ! CHECK: wr %i0, 5, %psr          ! encoding: [0x81,0x8e,0x20,0x05]
        wr %i0, 5, %psr

        ! CHECK: wr %i0, 5, %wim          ! encoding: [0x81,0x96,0x20,0x05]
        wr %i0, 5, %wim

        ! CHECK: wr %i0, 5, %tbr          ! encoding: [0x81,0x9e,0x20,0x05]
        wr %i0, 5, %tbr
