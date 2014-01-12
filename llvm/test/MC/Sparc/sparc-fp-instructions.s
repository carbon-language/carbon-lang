! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s

        ! CHECK: fitos %f0, %f4                  ! encoding: [0x89,0xa0,0x18,0x80]
        ! CHECK: fitod %f0, %f4                  ! encoding: [0x89,0xa0,0x19,0x00]
        ! CHECK: fitoq %f0, %f4                  ! encoding: [0x89,0xa0,0x19,0x80]
        fitos %f0, %f4
        fitod %f0, %f4
        fitoq %f0, %f4

        ! CHECK: fstoi %f0, %f4                  ! encoding: [0x89,0xa0,0x1a,0x20]
        ! CHECK: fdtoi %f0, %f4                  ! encoding: [0x89,0xa0,0x1a,0x40]
        ! CHECK: fqtoi %f0, %f4                  ! encoding: [0x89,0xa0,0x1a,0x60]
        fstoi %f0, %f4
        fdtoi %f0, %f4
        fqtoi %f0, %f4

        ! CHECK: fstod %f0, %f4                  ! encoding: [0x89,0xa0,0x19,0x20]
        ! CHECK: fstoq %f0, %f4                  ! encoding: [0x89,0xa0,0x19,0xa0]
        fstod %f0, %f4
        fstoq %f0, %f4

        ! CHECK: fdtos %f0, %f4                  ! encoding: [0x89,0xa0,0x18,0xc0]
        ! CHECK: fdtoq %f0, %f4                  ! encoding: [0x89,0xa0,0x19,0xc0]
        fdtos %f0, %f4
        fdtoq %f0, %f4

        ! CHECK: fqtos %f0, %f4                  ! encoding: [0x89,0xa0,0x18,0xe0]
        ! CHECK: fqtod %f0, %f4                  ! encoding: [0x89,0xa0,0x19,0x60]
        fqtos %f0, %f4
        fqtod %f0, %f4

        ! CHECK: fmovs %f0, %f4                  ! encoding: [0x89,0xa0,0x00,0x20]
        ! CHECK: fmovd %f0, %f4                  ! encoding: [0x89,0xa0,0x00,0x40]
        ! CHECK: fmovq %f0, %f4                  ! encoding: [0x89,0xa0,0x00,0x60]
        fmovs %f0, %f4
        fmovd %f0, %f4
        fmovq %f0, %f4

        ! CHECK: fnegs %f0, %f4                  ! encoding: [0x89,0xa0,0x00,0xa0]
        ! CHECK: fnegd %f0, %f4                  ! encoding: [0x89,0xa0,0x00,0xc0]
        ! CHECK: fnegq %f0, %f4                  ! encoding: [0x89,0xa0,0x00,0xe0]
        fnegs %f0, %f4
        fnegd %f0, %f4
        fnegq %f0, %f4

        ! CHECK: fabss %f0, %f4                  ! encoding: [0x89,0xa0,0x01,0x20]
        ! CHECK: fabsd %f0, %f4                  ! encoding: [0x89,0xa0,0x01,0x40]
        ! CHECK: fabsq %f0, %f4                  ! encoding: [0x89,0xa0,0x01,0x60]
        fabss %f0, %f4
        fabsd %f0, %f4
        fabsq %f0, %f4

        ! CHECK: fsqrts %f0, %f4                 ! encoding: [0x89,0xa0,0x05,0x20]
        ! CHECK: fsqrtd %f0, %f4                 ! encoding: [0x89,0xa0,0x05,0x40]
        ! CHECK: fsqrtq %f0, %f4                 ! encoding: [0x89,0xa0,0x05,0x60]
        fsqrts %f0, %f4
        fsqrtd %f0, %f4
        fsqrtq %f0, %f4

        ! CHECK: fadds %f0, %f4, %f8             ! encoding: [0x91,0xa0,0x08,0x24]
        ! CHECK: faddd %f0, %f4, %f8             ! encoding: [0x91,0xa0,0x08,0x44]
        ! CHECK: faddq %f0, %f4, %f8             ! encoding: [0x91,0xa0,0x08,0x64]
        fadds %f0, %f4, %f8
        faddd %f0, %f4, %f8
        faddq %f0, %f4, %f8

        ! CHECK: fsubs %f0, %f4, %f8             ! encoding: [0x91,0xa0,0x08,0xa4]
        ! CHECK: fsubd %f0, %f4, %f8             ! encoding: [0x91,0xa0,0x08,0xc4]
        ! CHECK: fsubq %f0, %f4, %f8             ! encoding: [0x91,0xa0,0x08,0xe4]
        fsubs %f0, %f4, %f8
        fsubd %f0, %f4, %f8
        fsubq %f0, %f4, %f8

        ! CHECK: fmuls %f0, %f4, %f8             ! encoding: [0x91,0xa0,0x09,0x24]
        ! CHECK: fmuld %f0, %f4, %f8             ! encoding: [0x91,0xa0,0x09,0x44]
        ! CHECK: fmulq %f0, %f4, %f8             ! encoding: [0x91,0xa0,0x09,0x64]
        fmuls %f0, %f4, %f8
        fmuld %f0, %f4, %f8
        fmulq %f0, %f4, %f8

        ! CHECK: fsmuld %f0, %f4, %f8            ! encoding: [0x91,0xa0,0x0d,0x24]
        ! CHECK: fdmulq %f0, %f4, %f8            ! encoding: [0x91,0xa0,0x0d,0xc4]
        fsmuld %f0, %f4, %f8
        fdmulq %f0, %f4, %f8

        ! CHECK: fdivs %f0, %f4, %f8             ! encoding: [0x91,0xa0,0x09,0xa4]
        ! CHECK: fdivd %f0, %f4, %f8             ! encoding: [0x91,0xa0,0x09,0xc4]
        ! CHECK: fdivq %f0, %f4, %f8             ! encoding: [0x91,0xa0,0x09,0xe4]
        fdivs %f0, %f4, %f8
        fdivd %f0, %f4, %f8
        fdivq %f0, %f4, %f8

        ! CHECK: fcmps %f0, %f4                  ! encoding: [0x81,0xa8,0x0a,0x24]
        ! CHECK: fcmpd %f0, %f4                  ! encoding: [0x81,0xa8,0x0a,0x44]
        ! CHECK: fcmpq %f0, %f4                  ! encoding: [0x81,0xa8,0x0a,0x64]
        fcmps %f0, %f4
        fcmpd %f0, %f4
        fcmpq %f0, %f4

        ! CHECK: fxtos %f0, %f4                  ! encoding: [0x89,0xa0,0x10,0x80]
        ! CHECK: fxtod %f0, %f4                  ! encoding: [0x89,0xa0,0x11,0x00]
        ! CHECK: fxtoq %f0, %f4                  ! encoding: [0x89,0xa0,0x11,0x80]
        fxtos %f0, %f4
        fxtod %f0, %f4
        fxtoq %f0, %f4

        ! CHECK: fstox %f0, %f4                  ! encoding: [0x89,0xa0,0x10,0x20]
        ! CHECK: fdtox %f0, %f4                  ! encoding: [0x89,0xa0,0x10,0x40]
        ! CHECK: fqtox %f0, %f4                  ! encoding: [0x89,0xa0,0x10,0x60]
        fstox %f0, %f4
        fdtox %f0, %f4
        fqtox %f0, %f4

