# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: mxdbr	%f0, %f0                # encoding: [0xb3,0x07,0x00,0x00]
#CHECK: mxdbr	%f0, %f15               # encoding: [0xb3,0x07,0x00,0x0f]
#CHECK: mxdbr	%f8, %f8                # encoding: [0xb3,0x07,0x00,0x88]
#CHECK: mxdbr	%f13, %f0               # encoding: [0xb3,0x07,0x00,0xd0]

	mxdbr	%f0, %f0
	mxdbr	%f0, %f15
	mxdbr	%f8, %f8
	mxdbr	%f13, %f0
