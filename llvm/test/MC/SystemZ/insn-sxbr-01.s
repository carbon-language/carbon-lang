# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sxbr	%f0, %f0                # encoding: [0xb3,0x4b,0x00,0x00]
#CHECK: sxbr	%f0, %f13               # encoding: [0xb3,0x4b,0x00,0x0d]
#CHECK: sxbr	%f8, %f8                # encoding: [0xb3,0x4b,0x00,0x88]
#CHECK: sxbr	%f13, %f0               # encoding: [0xb3,0x4b,0x00,0xd0]

	sxbr	%f0, %f0
	sxbr	%f0, %f13
	sxbr	%f8, %f8
	sxbr	%f13, %f0
