# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cgebr	%r0, 0, %f0             # encoding: [0xb3,0xa8,0x00,0x00]
#CHECK: cgebr	%r0, 0, %f15            # encoding: [0xb3,0xa8,0x00,0x0f]
#CHECK: cgebr	%r0, 15, %f0            # encoding: [0xb3,0xa8,0xf0,0x00]
#CHECK: cgebr	%r4, 5, %f6             # encoding: [0xb3,0xa8,0x50,0x46]
#CHECK: cgebr	%r15, 0, %f0            # encoding: [0xb3,0xa8,0x00,0xf0]

	cgebr	%r0, 0, %f0
	cgebr	%r0, 0, %f15
	cgebr	%r0, 15, %f0
	cgebr	%r4, 5, %f6
	cgebr	%r15, 0, %f0
