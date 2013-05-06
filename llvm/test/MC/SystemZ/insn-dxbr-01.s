# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: dxbr	%f0, %f0                # encoding: [0xb3,0x4d,0x00,0x00]
#CHECK: dxbr	%f0, %f13               # encoding: [0xb3,0x4d,0x00,0x0d]
#CHECK: dxbr	%f8, %f8                # encoding: [0xb3,0x4d,0x00,0x88]
#CHECK: dxbr	%f13, %f0               # encoding: [0xb3,0x4d,0x00,0xd0]

	dxbr	%f0, %f0
	dxbr	%f0, %f13
	dxbr	%f8, %f8
	dxbr	%f13, %f0
