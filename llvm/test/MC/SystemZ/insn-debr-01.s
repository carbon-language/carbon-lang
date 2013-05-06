# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: debr	%f0, %f0                # encoding: [0xb3,0x0d,0x00,0x00]
#CHECK: debr	%f0, %f15               # encoding: [0xb3,0x0d,0x00,0x0f]
#CHECK: debr	%f7, %f8                # encoding: [0xb3,0x0d,0x00,0x78]
#CHECK: debr	%f15, %f0               # encoding: [0xb3,0x0d,0x00,0xf0]

	debr	%f0, %f0
	debr	%f0, %f15
	debr	%f7, %f8
	debr	%f15, %f0
