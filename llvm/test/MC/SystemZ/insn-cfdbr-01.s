# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cfdbr	%r0, 0, %f0             # encoding: [0xb3,0x99,0x00,0x00]
#CHECK: cfdbr	%r0, 0, %f15            # encoding: [0xb3,0x99,0x00,0x0f]
#CHECK: cfdbr	%r0, 15, %f0            # encoding: [0xb3,0x99,0xf0,0x00]
#CHECK: cfdbr	%r4, 5, %f6             # encoding: [0xb3,0x99,0x50,0x46]
#CHECK: cfdbr	%r15, 0, %f0            # encoding: [0xb3,0x99,0x00,0xf0]

	cfdbr	%r0, 0, %f0
	cfdbr	%r0, 0, %f15
	cfdbr	%r0, 15, %f0
	cfdbr	%r4, 5, %f6
	cfdbr	%r15, 0, %f0
