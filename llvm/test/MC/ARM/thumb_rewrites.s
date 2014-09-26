@ RUN: llvm-mc -triple thumbv6m -show-encoding < %s | FileCheck %s

    adds    r0, r0, #8
@ CHECK: adds   r0, #8              @ encoding: [0x08,0x30]

    adds    r0, r0, r0
@ CHECK: adds   r0, r0, r0          @ encoding: [0x00,0x18]

    add     r0, r0, r8
@ CHECK: add    r0, r8              @ encoding: [0x40,0x44]

    add     sp, sp, r0
@ CHECK: add    sp, r0              @ encoding: [0x85,0x44]

    add     r0, r0, r1
@ CHECK: add    r0, r1              @ encoding: [0x08,0x44]

    add     r2, r2, r3
@ CHECK: add    r2, r3              @ encoding: [0x1a,0x44]

    subs    r0, r0, r0
@ CHECK: subs   r0, r0, r0          @ encoding: [0x00,0x1a]

    ands    r0, r0, r1
@ CHECK: ands   r0, r1              @ encoding: [0x08,0x40]

    eors    r0, r0, r1
@ CHECK: eors   r0, r1              @ encoding: [0x48,0x40]

    lsls    r0, r0, r1
@ CHECK: lsls   r0, r1              @ encoding: [0x88,0x40]

    lsrs    r0, r0, r1
@ CHECK: lsrs   r0, r1              @ encoding: [0xc8,0x40]

    asrs    r0, r0, r1
@ CHECK: asrs   r0, r1              @ encoding: [0x08,0x41]

    adcs    r0, r0, r1
@ CHECK: adcs   r0, r1              @ encoding: [0x48,0x41]

    sbcs    r0, r0, r1
@ CHECK: sbcs   r0, r1              @ encoding: [0x88,0x41]

    rors    r0, r0, r1
@ CHECK: rors   r0, r1              @ encoding: [0xc8,0x41]

    orrs    r0, r0, r1
@ CHECK: orrs   r0, r1              @ encoding: [0x08,0x43]

    bics    r0, r0, r1
@ CHECK: bics   r0, r1              @ encoding: [0x88,0x43]
