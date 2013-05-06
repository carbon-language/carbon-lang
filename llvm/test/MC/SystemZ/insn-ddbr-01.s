# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ddbr	%f0, %f0                # encoding: [0xb3,0x1d,0x00,0x00]
#CHECK: ddbr	%f0, %f15               # encoding: [0xb3,0x1d,0x00,0x0f]
#CHECK: ddbr	%f7, %f8                # encoding: [0xb3,0x1d,0x00,0x78]
#CHECK: ddbr	%f15, %f0               # encoding: [0xb3,0x1d,0x00,0xf0]

	ddbr	%f0, %f0
	ddbr	%f0, %f15
	ddbr	%f7, %f8
	ddbr	%f15, %f0
