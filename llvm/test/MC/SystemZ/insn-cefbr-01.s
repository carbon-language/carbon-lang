# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cefbr	%f0, %r0                # encoding: [0xb3,0x94,0x00,0x00]
#CHECK: cefbr	%f0, %r15               # encoding: [0xb3,0x94,0x00,0x0f]
#CHECK: cefbr	%f15, %r0               # encoding: [0xb3,0x94,0x00,0xf0]
#CHECK: cefbr	%f7, %r8                # encoding: [0xb3,0x94,0x00,0x78]
#CHECK: cefbr	%f15, %r15              # encoding: [0xb3,0x94,0x00,0xff]

	cefbr	%f0, %r0
	cefbr	%f0, %r15
	cefbr	%f15, %r0
	cefbr	%f7, %r8
	cefbr	%f15, %r15
