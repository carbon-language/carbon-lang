! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s

        ! CHECK: membar 15             ! encoding: [0x81,0x43,0xe0,0x0f]
        membar 15

        ! CHECK: cas [%i0], %l6, %o2   ! encoding: [0xd5,0xe6,0x10,0x16]
        cas [%i0], %l6, %o2

        ! CHECK: casx [%i0], %l6, %o2  ! encoding: [0xd5,0xf6,0x10,0x16]
        casx [%i0], %l6, %o2
