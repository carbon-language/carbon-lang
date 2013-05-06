# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cdgbr	%f0, %r0                # encoding: [0xb3,0xa5,0x00,0x00]
#CHECK: cdgbr	%f0, %r15               # encoding: [0xb3,0xa5,0x00,0x0f]
#CHECK: cdgbr	%f15, %r0               # encoding: [0xb3,0xa5,0x00,0xf0]
#CHECK: cdgbr	%f7, %r8                # encoding: [0xb3,0xa5,0x00,0x78]
#CHECK: cdgbr	%f15, %r15              # encoding: [0xb3,0xa5,0x00,0xff]

	cdgbr	%f0, %r0
	cdgbr	%f0, %r15
	cdgbr	%f15, %r0
	cdgbr	%f7, %r8
	cdgbr	%f15, %r15
