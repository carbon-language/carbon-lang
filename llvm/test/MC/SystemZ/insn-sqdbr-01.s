# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sqdbr	%f0, %f0                # encoding: [0xb3,0x15,0x00,0x00]
#CHECK: sqdbr	%f0, %f15               # encoding: [0xb3,0x15,0x00,0x0f]
#CHECK: sqdbr	%f7, %f8                # encoding: [0xb3,0x15,0x00,0x78]
#CHECK: sqdbr	%f15, %f0               # encoding: [0xb3,0x15,0x00,0xf0]

	sqdbr	%f0, %f0
	sqdbr	%f0, %f15
	sqdbr	%f7, %f8
	sqdbr	%f15, %f0
