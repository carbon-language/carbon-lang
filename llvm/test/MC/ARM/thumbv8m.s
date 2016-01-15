// RUN: not llvm-mc -triple=thumbv8m.base -show-encoding < %s 2>%t \
// RUN:   | FileCheck --check-prefix=CHECK %s
// RUN:     FileCheck --check-prefix=UNDEF-BASELINE --check-prefix=UNDEF < %t %s
// RUN: not llvm-mc -triple=thumbv8m.main -show-encoding < %s 2>%t \
// RUN:   | FileCheck --check-prefix=CHECK %s
// RUN:     FileCheck --check-prefix=UNDEF-MAINLINE --check-prefix=UNDEF < %t %s

// Simple check that baseline is v6M and mainline is v7M
// UNDEF-BASELINE: error: instruction requires: thumb2
// UNDEF-MAINLINE-NOT: error: instruction requires:
mov.w r0, r0

// Check that .arm is invalid
// UNDEF: target does not support ARM mode
.arm

// Instruction availibility checks

// 'Barrier instructions'

// CHECK: isb	sy              @ encoding: [0xbf,0xf3,0x6f,0x8f]
isb sy

// 'Code optimization'

// CHECK: sdiv r1, r2, r3     @ encoding: [0x92,0xfb,0xf3,0xf1]
sdiv r1, r2, r3

// CHECK: udiv r1, r2, r3     @ encoding: [0xb2,0xfb,0xf3,0xf1]
udiv r1, r2, r3

// 'XO generation'

// CHECK: movw r1, #65535            @ encoding: [0x4f,0xf6,0xff,0x71]
movw r1, #0xffff

// CHECK: movt r1, #65535            @ encoding: [0xcf,0xf6,0xff,0x71]
movt r1, #0xffff
