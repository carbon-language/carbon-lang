# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: adbr	%f0, %f0                # encoding: [0xb3,0x1a,0x00,0x00]
#CHECK: adbr	%f0, %f15               # encoding: [0xb3,0x1a,0x00,0x0f]
#CHECK: adbr	%f7, %f8                # encoding: [0xb3,0x1a,0x00,0x78]
#CHECK: adbr	%f15, %f0               # encoding: [0xb3,0x1a,0x00,0xf0]

	adbr	%f0, %f0
	adbr	%f0, %f15
	adbr	%f7, %f8
	adbr	%f15, %f0
