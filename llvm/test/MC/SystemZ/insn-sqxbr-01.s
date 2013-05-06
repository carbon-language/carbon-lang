# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sqxbr	%f0, %f0                # encoding: [0xb3,0x16,0x00,0x00]
#CHECK: sqxbr	%f0, %f13               # encoding: [0xb3,0x16,0x00,0x0d]
#CHECK: sqxbr	%f8, %f8                # encoding: [0xb3,0x16,0x00,0x88]
#CHECK: sqxbr	%f13, %f0               # encoding: [0xb3,0x16,0x00,0xd0]

	sqxbr	%f0, %f0
	sqxbr	%f0, %f13
	sqxbr	%f8, %f8
	sqxbr	%f13, %f0
