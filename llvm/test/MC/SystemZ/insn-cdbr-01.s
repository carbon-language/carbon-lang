# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cdbr	%f0, %f0                # encoding: [0xb3,0x19,0x00,0x00]
#CHECK: cdbr	%f0, %f15               # encoding: [0xb3,0x19,0x00,0x0f]
#CHECK: cdbr	%f7, %f8                # encoding: [0xb3,0x19,0x00,0x78]
#CHECK: cdbr	%f15, %f0               # encoding: [0xb3,0x19,0x00,0xf0]

	cdbr	%f0, %f0
	cdbr	%f0, %f15
	cdbr	%f7, %f8
	cdbr	%f15, %f0
