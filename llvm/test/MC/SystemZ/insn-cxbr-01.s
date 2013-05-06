# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cxbr	%f0, %f0                # encoding: [0xb3,0x49,0x00,0x00]
#CHECK: cxbr	%f0, %f13               # encoding: [0xb3,0x49,0x00,0x0d]
#CHECK: cxbr	%f8, %f8                # encoding: [0xb3,0x49,0x00,0x88]
#CHECK: cxbr	%f13, %f0               # encoding: [0xb3,0x49,0x00,0xd0]

	cxbr	%f0, %f0
	cxbr	%f0, %f13
	cxbr	%f8, %f8
	cxbr	%f13, %f0
