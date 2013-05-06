# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cdfbr	%f0, %r0                # encoding: [0xb3,0x95,0x00,0x00]
#CHECK: cdfbr	%f0, %r15               # encoding: [0xb3,0x95,0x00,0x0f]
#CHECK: cdfbr	%f15, %r0               # encoding: [0xb3,0x95,0x00,0xf0]
#CHECK: cdfbr	%f7, %r8                # encoding: [0xb3,0x95,0x00,0x78]
#CHECK: cdfbr	%f15, %r15              # encoding: [0xb3,0x95,0x00,0xff]

	cdfbr	%f0, %r0
	cdfbr	%f0, %r15
	cdfbr	%f15, %r0
	cdfbr	%f7, %r8
	cdfbr	%f15, %r15
