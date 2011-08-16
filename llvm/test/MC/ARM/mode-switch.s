@ Test ARM / Thumb mode switching with .code
@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: add.w	r0, r0, r1              @ encoding: [0x00,0xeb,0x01,0x00]
	add.w	r0, r0, r1

.code 32
@ CHECK: add	r0, r0, r1              @ encoding: [0x01,0x00,0x80,0xe0]
	add	r0, r0, r1

.code 16
@ CHECK: adds	r0, r0, r1              @ encoding: [0x40,0x18]

        adds    r0, r0, r1
