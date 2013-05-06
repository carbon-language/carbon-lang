# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: mxbr	%f0, %f0                # encoding: [0xb3,0x4c,0x00,0x00]
#CHECK: mxbr	%f0, %f13               # encoding: [0xb3,0x4c,0x00,0x0d]
#CHECK: mxbr	%f8, %f5                # encoding: [0xb3,0x4c,0x00,0x85]
#CHECK: mxbr	%f13, %f13              # encoding: [0xb3,0x4c,0x00,0xdd]

	mxbr	%f0, %f0
	mxbr	%f0, %f13
	mxbr	%f8, %f5
	mxbr	%f13, %f13
