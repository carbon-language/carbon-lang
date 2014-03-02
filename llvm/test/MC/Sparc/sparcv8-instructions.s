! RUN: llvm-mc %s -arch=sparc -show-encoding | FileCheck %s

        ! CHECK: fcmps %f0, %f4           ! encoding: [0x81,0xa8,0x0a,0x24]
        ! CHECK: fcmpd %f0, %f4           ! encoding: [0x81,0xa8,0x0a,0x44]
        ! CHECK: fcmpq %f0, %f4           ! encoding: [0x81,0xa8,0x0a,0x64]
        fcmps %f0, %f4
        fcmpd %f0, %f4
        fcmpq %f0, %f4
