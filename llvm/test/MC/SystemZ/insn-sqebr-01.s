# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sqebr	%f0, %f0                # encoding: [0xb3,0x14,0x00,0x00]
#CHECK: sqebr	%f0, %f15               # encoding: [0xb3,0x14,0x00,0x0f]
#CHECK: sqebr	%f7, %f8                # encoding: [0xb3,0x14,0x00,0x78]
#CHECK: sqebr	%f15, %f0               # encoding: [0xb3,0x14,0x00,0xf0]

	sqebr	%f0, %f0
	sqebr	%f0, %f15
	sqebr	%f7, %f8
	sqebr	%f15, %f0
